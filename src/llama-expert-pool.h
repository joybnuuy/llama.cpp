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

        // Sentence-local accumulators
        uint64_t sent_tokens                = 0;
        uint64_t sent_free_pass_tokens      = 0;
        uint64_t sent_constrained_tokens    = 0;
        uint64_t sent_unique_experts        = 0;  // union across all layers
        uint64_t sent_pool_hits             = 0;  // constrained selections that map to a filled pool slot
        uint64_t sent_pool_misses           = 0;  // constrained selections mapping to -1 (should never happen)

        std::vector<std::unordered_set<int32_t>> sent_layer_experts; // per-layer unique experts
        std::vector<std::vector<uint64_t>>       sent_slot_hits;     // per-layer per-slot hit count

        void start_sentence(int n_layer, int pool_size) {
            sent_tokens             = 0;
            sent_free_pass_tokens   = 0;
            sent_constrained_tokens = 0;
            sent_unique_experts     = 0;
            sent_pool_hits          = 0;
            sent_pool_misses        = 0;
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

        void end_sentence() {
            total_sentences++;
            total_refreshes++;
            total_pool_hits   += sent_pool_hits;
            total_pool_misses += sent_pool_misses;
            uint64_t total_unique = 0;
            for (const auto & s : sent_layer_experts) {
                total_unique += s.size();
            }
            sent_unique_experts = total_unique;
        }
    } stats;

    bool init(int pool_size, int n_expert, int n_layer, int n_embd, int n_ff_exp,
              ggml_backend_buffer_type_t buft,
              ggml_type type_gate_inp,
              ggml_type type_up_exps,
              ggml_type type_gate_exps,
              ggml_type type_down_exps,
              ggml_type type_gate_up_exps);
    bool is_constrained() const { return state == CONSTRAINED && pool_size > 0; }
    bool is_free_pass() const { return state == FREE_PASS && pool_size > 0; }

    // Refresh pool for a layer: copy expert data from full tensor to pooled tensor
    void refresh_layer(int il, const llama_layer & layer, const std::vector<int32_t> & new_pool);

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
