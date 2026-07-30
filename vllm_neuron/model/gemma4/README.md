# Gemma 4 native model port

This package is the native vLLM-Neuron implementation boundary. It must use
the same runner-facing contracts as `llama3` and `gpt_oss` in this repository:

- `forward()` receives vLLM's padded inputs and attention metadata;
- the model writes K/V through the paged cache using `slot_mapping`;
- `sampling_positions` selects only real-token logits;
- TP/EP sharding uses vLLM-Neuron parallel layers and collectives;
- all graph-affecting shapes are represented by `NeuronConfig` buckets.

The Gemma 4 architecture has two attention families (sliding/local and global)
with different head widths and a sparse MoE feed-forward block. The native
port must preserve those per-layer shapes instead of normalizing them to a
single cache layout.

Porting order:

1. config and layer metadata;
2. RMS/value normalization and rotary embeddings;
3. local/global attention and paged-KV writes;
4. MoE router and expert parallelism;
5. embeddings/lm-head and sampling positions;
6. weight loading and direct ModelRegistry smoke test;
7. greedy token matching against the baseline and performance benchmark.
