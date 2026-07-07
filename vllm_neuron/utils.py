# SPDX-License-Identifier: Apache-2.0
import logging
import os

logger = logging.getLogger(__name__)

_BASE_PORT = 62182


def set_unique_rt_root_comm_id():
    """Set a unique NEURON_RT_ROOT_COMM_ID based on visible cores.

    Checks NEURON_RT_VISIBLE_CORES to derive a deterministic port
    offset from the first core ID.

    Port scheme: 62182 + first_core_id
      - cores 0-7    → port 62182
      - cores 16-31  → port 62198

    Safe because core sets never overlap within a single node.
    """
    visible = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
    if not visible:
        logger.debug("no core visibility set — no-op")
        return

    try:
        first_device = int(visible.split(",")[0].split("-")[0])
    except (ValueError, IndexError):
        logger.warning(
            "failed to parse first core from visible=%r — no-op",
            visible,
        )
        return

    port = _BASE_PORT + first_device
    root_addr = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["NEURON_RT_ROOT_COMM_ID"] = f"{root_addr}:{port}"
    logger.debug(
        "set NEURON_RT_ROOT_COMM_ID=%s (base_port=%d + first_core=%d)",
        os.environ["NEURON_RT_ROOT_COMM_ID"],
        _BASE_PORT,
        first_device,
    )
