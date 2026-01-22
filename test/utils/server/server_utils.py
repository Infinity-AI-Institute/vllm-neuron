# SPDX-License-Identifier: Apache-2.0
"""Server utilities for vLLM tests."""

import logging
import signal
import time

import psutil
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def kill_process_and_children(pid: int):
    """Kill process and all children"""
    try:
        parent = psutil.Process(pid)
        procs = [parent] + parent.children(recursive=True)
        for p in procs:
            p.send_signal(signal.SIGTERM)
        _, alive = psutil.wait_procs(procs, timeout=30)
        for p in alive:
            p.send_signal(signal.SIGKILL)
    except Exception as e:
        logger.warning(f"Failed to terminate pid={pid}: {e}")


def kill_children_of_process_on_port(port: int):
    """Kill processes listening on port"""
    pids = {
        c.pid
        for c in psutil.net_connections()
        if getattr(c, "laddr", None)
        and getattr(c.laddr, "port", None) == port
        and c.pid
    }
    for pid in pids:
        kill_process_and_children(pid)


def wait_for_server(port: int, timeout: int = 600) -> bool:
    """Wait for server to become healthy"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False
