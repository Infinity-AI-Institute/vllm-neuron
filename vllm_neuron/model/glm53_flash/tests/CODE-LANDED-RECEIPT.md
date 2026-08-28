# GLM-5.3-Flash source-port receipt

- Recorded: `2026-08-27T20:55:48-04:00`
- Branch: `codex/glm53-flash-enablement`
- Starting HEAD: `320f82847942a87dcc5d1e770c593947e44b6417`
- Scope: local source and CPU-reference gates only; no device compile, push, or
  operator-owned model directory.

## Results

Required config and dispatch tests:

```text
uv run --no-project --with pytest --with torch --with numpy python -B -m pytest -q \
  C:/Users/apumu/research/InfinityAI/vllm-neuron-codex-alpha/vllm_neuron/model/glm53_flash/tests/test_config_load.py \
  C:/Users/apumu/research/InfinityAI/vllm-neuron-codex-alpha/vllm_neuron/model/glm53_flash/tests/test_layer_dispatch.py

7 passed, 1 skipped in 3.17s
```

The skipped case is the opt-in live Hugging Face check. It was run separately:

```text
GLM53_RUN_HF_CONFIG_TEST=1 ... pytest -q \
  C:/Users/apumu/research/InfinityAI/vllm-neuron-codex-alpha/vllm_neuron/model/glm53_flash/tests/test_config_load.py
2 passed in 7.57s
```

The complete source-reference suite, including all four Fleet-A import
contracts, graph/cache ABI identity, FP8 rejection, mHC Sinkhorn, and
prefill-to-greedy-decode state continuity:

```text
uv run --no-project --with pytest --with torch --with numpy python -B -m pytest -q \
  C:/Users/apumu/research/InfinityAI/vllm-neuron-codex-alpha/vllm_neuron/model/glm53_flash/tests

10 passed, 1 skipped in 2.90s
```

Static quality gates:

```text
ruff check: All checks passed
ruff format --check: 19 files already formatted
git diff --check: PASS
official-config audit: 47/47 text numeric-or-boolean fields captured;
                       4/4 linear-attention numeric-or-boolean fields captured
```

Staged ten-token evaluator:

```text
uv run --no-project --with pytest python -B -m pytest -q \
  C:/Users/apumu/research/InfinityAI/gemma4-trn2-handoff/harness-v2/staging/reference-sweep-20260826T2150Z/lanes/glm-5-2-5-3/tests/test_06_ten_token_exact_gate.py
5 passed, 1 skipped in 0.22s
```

The live ten-token case is not claimed as a pass. At receipt time,
`C:/Users/apumu/research/InfinityAI/gemma4-trn2-handoff/harness-v2/staging/reference-sweep-20260826T2150Z/lanes/glm-5-2-5-3/reference-logits-glm53-flash/`
did not exist and `GLM_TEN_TOKEN_ARTIFACT`
could not be set without fabricating an oracle. Coordination is tracked at
`https://github.com/Infinity-AI-Institute/trainium-autoresearcher/issues/45`.
