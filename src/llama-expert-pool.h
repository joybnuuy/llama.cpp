#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>
#include <cstdint>
#include <memory>
#include <unordered_set>

struct llama_layer;

// MoE expert pool for B2 VRAM-saving sentence-level expert caching
struct llama_expert_pool {
    struct layer_pool {
        ggml_tensor * gate_inp     = nullptr;  // [n_embd, pool_size]
        ggml_tensor * up_exps      = nullptr;  // [n_embd, n_ff_exp, pool_size]
        ggml_tensor * gate_exps    = nullptr;  // [n_embd, n_ff_exp, pool_size]
        ggml_tensor * down_exps    = nullptr;  // [n_ff_exp, n_embd, pool_size]
        ggml_tensor * gate_up_exps = nullptr;  // fused gate+up [n_embd*2, n_ff_exp, pool_size]
    };

    int pool_size  = 0;
    int n_expert   = 0;
    int n_layer    = 0;
    int n_embd     = 0;
    int n_ff_exp   = 0;

    std::vector<layer_pool> layers;

    // Current pool per layer: pool[il][slot] = global_expert_id
    std::vector<std::vector<int32_t>> pool;

    // Accumulated expert scores during bootstrap (reset per sentence)
    std::vector<std::vector<float>> expert_scores;

    enum state { FREE_PASS, CONSTRAINED } state = FREE_PASS;

    // Number of bootstrap (free-pass) tokens at sentence start
    int bootstrap_n = 3;
    int free_pass_remaining = 0;

    // Backfill: max experts to copy per token in constrained mode
    int refresh_budget = 5;

    // Token counter for LRU tracking
    uint64_t token_counter = 0;

    // Per-layer per-slot last-used timestamp (for LRU eviction)
    std::vector<std::vector<uint64_t>> slot_last_used;

    // Incremented when pool changes; used for graph reuse invalidation
    uint64_t pool_generation = 0;

    // GGML context for pool tensor allocation
    using ggml_context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    ggml_context_ptr ctx{nullptr, &ggml_free};

    // Backend buffer for pool tensors
    using ggml_backend_buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    ggml_backend_buffer_ptr buf{nullptr, &ggml_backend_buffer_free};

    // Debug / stats
    struct stats {
        uint64_t total_free_pass_tokens     = 0;
        uint64_t total_constrained_tokens   = 0;
        uint64_t total_sentences            = 0;
        uint64_t total_refreshes            = 0;
        uint64_t total_pool_hits            = 0;  // cumulative across all sentences
        uint64_t total_pool_misses          = 0;  // cumulative across all sentences

        // TRUE hit rate: full-router top-k selections that are already in pool
        uint64_t total_true_hits            = 0;
        uint64_t total_true_misses          = 0;

        // Sentence-local accumulators
        uint64_t sent_tokens                = 0;
        uint64_t sent_free_pass_tokens      = 0;
        uint64_t sent_constrained_tokens    = 0;
        uint64_t sent_unique_experts        = 0;  // union across all layers
        uint64_t sent_pool_hits             = 0;  // constrained selections that map to a filled pool slot
        uint64_t sent_pool_misses           = 0;  // constrained selections mapping to -1 (should never happen)
        uint64_t sent_true_hits             = 0;  // full-router selections present in pool
        uint64_t sent_true_misses           = 0;  // full-router selections NOT present in pool

        std::vector<std::unordered_set<int32_t>> sent_layer_experts; // per-layer unique experts
        std::vector<std::vector<uint64_t>>       sent_slot_hits;     // per-layer per-slot hit count

        void start_sentence(int n_layer, int pool_size) {
            sent_tokens             = 0;
            sent_free_pass_tokens   = 0;
            sent_constrained_tokens = 0;
            sent_unique_experts     = 0;
            sent_pool_hits          = 0;
            sent_pool_misses        = 0;
            sent_true_hits          = 0;
            sent_true_misses        = 0;
            sent_layer_experts.assign(n_layer, {});
            sent_slot_hits.assign(n_layer, std::vector<uint64_t>(pool_size, 0));
        }

        void record_token(bool free_pass, int n_layer,
                          const std::vector<std::vector<int32_t>> & selected_experts,
                          const std::vector<std::vector<int32_t>> & pool_mapping) {
            sent_tokens++;
            if (free_pass) {
                sent_free_pass_tokens++;
                total_free_pass_tokens++;
            } else {
                sent_constrained_tokens++;
                total_constrained_tokens++;
            }
            for (int il = 0; il < n_layer && il < (int)selected_experts.size(); ++il) {
                for (int32_t eid : selected_experts[il]) {
                    if (eid < 0) continue;
                    if (free_pass) {
                        sent_layer_experts[il].insert(eid);
                    } else {
                        // eid is a pool slot; map back to global expert
                        if (il < (int)pool_mapping.size() && eid < (int)pool_mapping[il].size()) {
                            int32_t global_eid = pool_mapping[il][eid];
                            if (global_eid >= 0) {
                                sent_layer_experts[il].insert(global_eid);
                                sent_pool_hits++;
                                if (eid < (int)sent_slot_hits[il].size()) {
                                    sent_slot_hits[il][eid]++;
                                }
                            } else {
                                sent_pool_misses++;
                            }
                        }
                    }
                }
            }
        }

        void record_true_hits(int n_layer,
                              const std::vector<std::vector<int32_t>> & full_topk,
                              const std::vector<std::vector<int32_t>> & pool_mapping) {
            // Build a per-layer set of experts currently in the pool for fast lookup
            for (int il = 0; il < n_layer && il < (int)full_topk.size(); ++il) {
                std::unordered_set<int32_t> pool_set;
                if (il < (int)pool_mapping.size()) {
                    for (int32_t eid : pool_mapping[il]) {
                        if (eid >= 0) pool_set.insert(eid);
                    }
                }
                for (int32_t eid : full_topk[il]) {
                    if (eid < 0) continue;
                    if (pool_set.count(eid)) {
                        sent_true_hits++;
                    } else {
                        sent_true_misses++;
                    }
                }
            }
        }

        void end_sentence() {
            total_sentences++;
            total_refreshes++;
            total_pool_hits   += sent_pool_hits;
            total_pool_misses += sent_pool_misses;
            total_true_hits   += sent_true_hits;
            total_true_misses += sent_true_misses;
            uint64_t total_unique = 0;
            for (const auto & s : sent_layer_experts) {
                total_unique += s.size();
            }
            sent_unique_experts = total_unique;
        }
    } stats;

    // Per-layer quantization types (models with mixed quantization have different types per layer)
    struct layer_types {
        ggml_type gate_inp     = GGML_TYPE_F32;
        ggml_type up_exps      = GGML_TYPE_F32;
        ggml_type gate_exps    = GGML_TYPE_F32;
        ggml_type down_exps    = GGML_TYPE_F32;
        ggml_type gate_up_exps = GGML_TYPE_F32;
    };

    bool init(int pool_size, int n_expert, int n_layer, int n_embd, int n_ff_exp,
              ggml_backend_buffer_type_t buft,
              const std::vector<layer_types> & types_per_layer,
              bool has_gate_up_exps);
    bool is_constrained() const { return state == CONSTRAINED && pool_size > 0; }
    bool is_free_pass() const { return state == FREE_PASS && pool_size > 0; }

    // Refresh pool for a layer: copy expert data from full tensor to pooled tensor
    void refresh_layer(int il, const llama_layer & layer, const std::vector<int32_t> & new_pool);

    // Refresh a single slot in a layer: copy one expert into one pool slot
    void refresh_slot(int il, const llama_layer & layer, int32_t slot, int32_t eid);

    // Reset expert scores (called at sentence boundary before bootstrap)
    void reset_scores();

    // Accumulate selected experts into scores (flat count, no decay)
    void accumulate_experts(const std::vector<std::vector<int32_t>> & selected_experts);

    // Select top pool_size experts from accumulated scores
    void finalize_pool();

    // Legacy: compute pool with EMA decay (kept for potential future use)
    void compute_new_pool(const std::vector<std::vector<int32_t>> & selected_experts);

    // Get effective n_expert for graph building
    int get_n_expert() const { return is_constrained() ? pool_size : n_expert; }
};
