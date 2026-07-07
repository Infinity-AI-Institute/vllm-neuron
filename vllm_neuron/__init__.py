# SPDX-License-Identifier: Apache-2.0
"""VllmNeuronPlugin module."""

import glob
import warnings
from vllm_neuron.utils import set_unique_rt_root_comm_id


def _is_neuron_dev() -> bool:
    """Detect Neuron device by checking for /dev/neuron* devices."""
    neuron_devices = glob.glob("/dev/neuron*")
    return len(neuron_devices) > 0


def register():
    """Register the Neuron platform if Neuron devices are present, else return None."""
    if not _is_neuron_dev():
        warnings.warn(
            "No Neuron devices found. Skipping Neuron plugin registration.",
            category=UserWarning,
        )
        return None
    return "vllm_neuron.platform.NeuronPlatform"


# Set unique CCOM bootstrap port to overwrite hard-coded NEURON_RT_ROOT_COMM_ID in XLA based torch_neuronx,
# to prevent port collisions in DP and DI configurations.
set_unique_rt_root_comm_id()
