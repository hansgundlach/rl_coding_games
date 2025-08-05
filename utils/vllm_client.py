#!/usr/bin/env python3
"""
vLLM Client Integration for GRPO Code Execution Training
Optimized for Supercloud V100s with offline support
"""

import os
import sys
import time
import logging
import subprocess
import requests
import json
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import threading
import signal

logger = logging.getLogger(__name__)


@dataclass
class VLLMConfig:
    """vLLM configuration loaded from YAML config"""

    enabled: bool
    server: Dict[str, Any]
    client: Dict[str, Any]
    generation: Dict[str, Any]
    integration: Dict[str, Any]


class VLLMServerManager:
    """Manages vLLM server lifecycle"""

    def __init__(self, config: VLLMConfig, model_path: str, offline_mode: bool = True):
        self.config = config
        self.model_path = model_path
        self.offline_mode = offline_mode
        self.server_process = None
        self.is_running = False

    def start_server(self) -> bool:
        """Start vLLM server with offline support"""
        if not self.config.enabled:
            logger.info("vLLM disabled in config, skipping server start")
            return False

        if self.is_server_running():
            logger.info("vLLM server already running")
            return True

        try:
            # Set offline environment variables for Supercloud
            env = os.environ.copy()
            if self.offline_mode:
                env.update(
                    {
                        "TRANSFORMERS_OFFLINE": "1",
                        "HF_HUB_OFFLINE": "1",
                        "HF_DATASETS_OFFLINE": "1",
                        "VLLM_DO_NOT_TRACK": "1",
                    }
                )

            # Build vLLM server command
            cmd = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                self.model_path,
                "--host",
                self.config.server["host"],
                "--port",
                str(self.config.server["port"]),
                "--gpu-memory-utilization",
                str(self.config.server["gpu_memory_utilization"]),
                "--max-model-len",
                str(self.config.server["max_model_len"]),
                "--tensor-parallel-size",
                str(self.config.server["tensor_parallel_size"]),
                "--max-num-batched-tokens",
                str(self.config.server["max_num_batched_tokens"]),
                "--max-num-seqs",
                str(self.config.server["max_num_seqs"]),
                "--swap-space",
                str(self.config.server["swap_space"]),
                "--dtype",
                self.config.server["dtype"],
            ]

            # Add offline mode flags
            if self.config.server.get("offline_mode", True):
                (
                    cmd.extend(["--disable-log-requests"])
                    if self.config.server.get("disable_log_requests")
                    else None
                )

            if self.config.server.get("trust_remote_code"):
                cmd.append("--trust-remote-code")

            logger.info(f"🚀 Starting vLLM server: {' '.join(cmd)}")

            # Start server process
            self.server_process = subprocess.Popen(
                cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Wait for server to be ready
            if self._wait_for_server():
                self.is_running = True
                logger.info("✅ vLLM server started successfully")
                return True
            else:
                logger.error("❌ vLLM server failed to start")
                self.stop_server()
                return False

        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            return False

    def _wait_for_server(self) -> bool:
        """Wait for server to be ready"""
        base_url = self.config.client["base_url"]
        timeout = self.config.integration["wait_for_server"]

        for i in range(timeout):
            try:
                response = requests.get(f"{base_url}/health", timeout=1)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
            if i % 5 == 0 and i > 0:
                logger.info(f"⏳ Waiting for vLLM server... ({i}/{timeout}s)")

        return False

    def is_server_running(self) -> bool:
        """Check if server is running"""
        if not self.config.enabled:
            return False

        try:
            base_url = self.config.client["base_url"]
            response = requests.get(f"{base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def stop_server(self):
        """Stop vLLM server"""
        if self.server_process:
            logger.info("🛑 Stopping vLLM server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            self.server_process = None
            self.is_running = False


class VLLMClient:
    """OpenAI-compatible client for vLLM server"""

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.base_url = config.client["base_url"]
        self.timeout = config.client["timeout"]
        self.max_retries = config.client["max_retries"]

    def generate_completions(
        self, prompts: Union[str, List[str]], mode: str = "grpo_completions"
    ) -> List[str]:
        """Generate completions using vLLM server"""
        if isinstance(prompts, str):
            prompts = [prompts]

        generation_config = self.config.generation.get(
            mode, self.config.generation["grpo_completions"]
        )

        completions = []

        for prompt in prompts:
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        f"{self.base_url}/completions",
                        json={
                            "model": "served-model",  # vLLM uses this placeholder
                            "prompt": prompt,
                            "max_tokens": generation_config["max_tokens"],
                            "temperature": generation_config["temperature"],
                            "top_p": generation_config["top_p"],
                            "frequency_penalty": generation_config["frequency_penalty"],
                            "presence_penalty": generation_config["presence_penalty"],
                            "stop": generation_config["stop"],
                        },
                        timeout=self.timeout,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        completion = result["choices"][0]["text"]
                        completions.append(completion)
                        break
                    else:
                        logger.warning(
                            f"vLLM request failed (attempt {attempt+1}): {response.status_code}"
                        )

                except Exception as e:
                    logger.warning(f"vLLM request error (attempt {attempt+1}): {e}")

                if attempt == self.max_retries - 1:
                    logger.error(f"vLLM failed after {self.max_retries} attempts")
                    completions.append("")  # Empty completion as fallback

        return completions


class VLLMIntegration:
    """Main vLLM integration class"""

    def __init__(
        self, config: Dict[str, Any], model_path: str, offline_mode: bool = True
    ):
        self.vllm_config = VLLMConfig(
            enabled=config.get("enabled", False),
            server=config.get("server", {}),
            client=config.get("client", {}),
            generation=config.get("generation", {}),
            integration=config.get("integration", {}),
        )

        self.model_path = model_path
        self.offline_mode = offline_mode
        self.server_manager = None
        self.client = None
        self.fallback_to_hf = self.vllm_config.integration.get("fallback_to_hf", True)

        if self.vllm_config.enabled:
            self.server_manager = VLLMServerManager(
                self.vllm_config, model_path, offline_mode
            )
            self.client = VLLMClient(self.vllm_config)

    def initialize(self) -> bool:
        """Initialize vLLM integration"""
        if not self.vllm_config.enabled:
            logger.info("vLLM integration disabled")
            return False

        if self.vllm_config.integration.get("auto_start_server", True):
            success = self.server_manager.start_server()
            if not success and not self.fallback_to_hf:
                raise RuntimeError("vLLM server failed to start and fallback disabled")
            return success
        else:
            return self.server_manager.is_server_running()

    def generate_completions(
        self, prompts: Union[str, List[str]], mode: str = "grpo_completions"
    ) -> List[str]:
        """Generate completions, with HF fallback if configured"""
        if not self.vllm_config.enabled or not self.server_manager.is_server_running():
            if self.fallback_to_hf:
                logger.warning("vLLM not available, falling back to HuggingFace")
                return []  # Will be handled by calling code
            else:
                raise RuntimeError("vLLM not available and fallback disabled")

        return self.client.generate_completions(prompts, mode)

    def cleanup(self):
        """Cleanup vLLM resources"""
        if self.server_manager:
            self.server_manager.stop_server()


# Global vLLM integration instance
_vllm_integration = None


def get_vllm_integration() -> Optional[VLLMIntegration]:
    """Get global vLLM integration instance"""
    return _vllm_integration


def initialize_vllm_integration(
    config: Dict[str, Any], model_path: str, offline_mode: bool = True
) -> VLLMIntegration:
    """Initialize global vLLM integration"""
    global _vllm_integration
    _vllm_integration = VLLMIntegration(config, model_path, offline_mode)
    return _vllm_integration


def cleanup_vllm_integration():
    """Cleanup global vLLM integration"""
    global _vllm_integration
    if _vllm_integration:
        _vllm_integration.cleanup()
        _vllm_integration = None
