"""Native Gemma 4 model implementation scaffold.

This module is the path-2 porting seam. It must own vLLM's paged KV-cache
writes and sampling-position contract; it must not route through NxDI model
registries or architecture-rewrite shims.
"""

import torch.nn as nn


class Gemma4MoeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        raise NotImplementedError(
            "Gemma4 native vLLM-Neuron layers are not implemented yet; "
            "use the committed serving baseline while this port is developed."
        )
