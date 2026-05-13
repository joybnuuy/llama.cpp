# Bug: Stale LRU Timestamps Cause Bad Evictions in Backfill

**Status:** Fixed (uncommitted, on moe-expert-pool branch)

## Symptom

With `--moe-pool-refresh-budget > 0`, backfill evicts wrong experts, causing model output to degrade (repetition, wrong answers). With backfill disabled (`budget=0`), output is correct.

## Root Cause

In constrained mode, `read_back_experts()` reads from `ffn_moe_topk-{il}` — a tensor that the CUDA fused `topk_moe` kernel does **not** write. It contains stale global expert IDs from the free-pass phase (values 0–255), not pool-local slot indices (0–79).

The old LRU update code:
```cpp
for (int32_t slot : selected_experts[il]) {
    if (slot >= 0 && slot < expert_pool.pool_size) {  // ALWAYS FALSE for global IDs > pool_size
        slot_last_used[il][slot] = token_counter;
    }
}
```

All timestamps stay at 0. LRU eviction always picks the same slot(s), repeatedly evicting critical experts.

## Fix

Use `full_topk` (shadow full-router path, which IS correctly written by CUDA) and map global expert IDs back to pool slots:

```cpp
for (int32_t eid : full_topk[il]) {
    if (eid < 0) continue;
    for (int s = 0; s < expert_pool.pool_size; ++s) {
        if (expert_pool.pool[il][s] == eid) {
            slot_last_used[il][s] = token_counter;
            break;
        }
    }
}
```

## Related Known Issue

`ffn_moe_topk-{il}` stale data is the same root cause as the "hit_rate stat always 0%" issue. The fused CUDA kernel handles softmax+argsort+get_rows in one pass and doesn't write the intermediate argsort result to the tensor. Both bugs stem from the same missing write.

## Files Changed

- `src/llama-context.cpp` — constrained mode LRU update logic
