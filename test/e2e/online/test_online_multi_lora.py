# SPDX-License-Identifier: Apache-2.0
import logging
from test.e2e.online.configs import MULTI_LORA_CONFIGS
from test.e2e.online.online_server_runner import run_online_integration

import pytest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "cfg", MULTI_LORA_CONFIGS, ids=[c.name for c in MULTI_LORA_CONFIGS]
)
def test_online_integration(cfg):
    """Parametrized integration tests"""
    logger.info("Running integration test with config: %s", cfg.name)
    run_online_integration(cfg)
