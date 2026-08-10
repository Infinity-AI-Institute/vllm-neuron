import torch
import operator

from vllm_neuron.fx_passes.inplace_rewrite_pass import InPlaceToOutOfPlacePass

def test_dynamic_paged_write_preserves_fixed_dimension():
    fx_graph = torch.fx.Graph()
    state = fx_graph.placeholder("state")
    blocks = fx_graph.placeholder("blocks")
    offsets = fx_graph.placeholder("offsets")
    values = fx_graph.placeholder("values")
    fx_graph.call_function(
        operator.setitem,
        args=(state, (blocks, 0, offsets, slice(None)), values),
    )
    fx_graph.output(state)
    graph = torch.fx.GraphModule(torch.nn.Module(), fx_graph)
    graph, _ = InPlaceToOutOfPlacePass().run(graph)

    state = torch.zeros((4, 1, 8, 3))
    blocks = torch.tensor([1, 3], dtype=torch.long)
    offsets = torch.tensor([2, 5], dtype=torch.long)
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    actual = graph(state, blocks, offsets, values)

    expected = state.clone()
    expected[1, 0, 2, :] = values[0]
    expected[3, 0, 5, :] = values[1]
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[:, 0]) == 6
    assert torch.count_nonzero(actual[:, 0, 0]) == 0
