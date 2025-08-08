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

    def _resolve_offline_model_path(self, model_path: str) -> str:
        """Resolve a local snapshot path for a HF model id when offline.

        Tries TRANSFORMERS_CACHE, HF_HOME, then ~/.cache/huggingface/hub.
        Falls back to the provided model_path if nothing is found.
        """
        try:
            if os.path.isdir(model_path):
                return model_path

            # Derive cache roots
            transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
            hf_home = os.environ.get("HF_HOME")
            default_hub = os.path.expanduser("~/.cache/huggingface/hub")

            cache_roots = [p for p in [transformers_cache, hf_home, default_hub] if p]

            # Convert model id to cache dir name
            safe_model = model_path.replace("/", "--")
            for root in cache_roots:
                candidate = os.path.join(root, f"models--{safe_model}", "snapshots")
                if os.path.isdir(candidate):
                    # choose the most recent snapshot
                    snapshots = [
                        os.path.join(candidate, d)
                        for d in os.listdir(candidate)
                        if os.path.isdir(os.path.join(candidate, d))
                    ]
                    if snapshots:
                        snapshots.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                        return snapshots[0]
        except Exception:
            pass

        return model_path

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
            model_arg = (
                self._resolve_offline_model_path(self.model_path)
                if self.offline_mode
                else self.model_path
            )

            server_cfg = self.config.server
            # sensible defaults for V100s / Supercloud
            dtype = server_cfg.get("dtype", "float16")
            swap_space = str(server_cfg.get("swap_space", 8))  # in GB
            gpu_util = str(server_cfg.get("gpu_memory_utilization", 0.7))
            max_len = str(server_cfg.get("max_model_len", 2048))
            tp_size = str(server_cfg.get("tensor_parallel_size", 1))
            max_batched_tokens = str(server_cfg.get("max_num_batched_tokens", 2048))
            max_num_seqs = str(server_cfg.get("max_num_seqs", 32))
            host = server_cfg.get("host", "127.0.0.1")
            port = str(server_cfg.get("port", 8000))

            cmd = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model_arg,
                "--host",
                host,
                "--port",
                port,
                "--gpu-memory-utilization",
                gpu_util,
                "--max-model-len",
                max_len,
                "--tensor-parallel-size",
                tp_size,
                "--max-num-batched-tokens",
                max_batched_tokens,
                "--max-num-seqs",
                max_num_seqs,
                "--swap-space",
                swap_space,
                "--dtype",
                dtype,
                "--served-model-name",
                "served-model",
                "--tokenizer-mode",
                "auto",
            ]

            # Add offline mode flags
            if self.config.server.get("offline_mode", True) and self.config.server.get(
                "disable_log_requests", True
            ):
                cmd.extend(["--disable-log-requests"])

            if self.config.server.get("trust_remote_code", True):
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
                # Capture and print stderr for debugging
                if self.server_process and self.server_process.stderr:
                    stderr_output = self.server_process.stderr.read()
                    if stderr_output:
                        logger.error(f"vLLM server stderr:\n{stderr_output}")
                self.stop_server()
                return False

        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            return False

    def _wait_for_server(self) -> bool:
        """Wait for server to be ready"""
        base_url = self.config.client.get("base_url", "http://127.0.0.1:8000/v1")
        # Wait up to 90s by default for cold start on V100s
        timeout = int(self.config.integration.get("wait_for_server", 90))

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
            base_url = self.config.client.get("base_url", "http://127.0.0.1:8000/v1")
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
        self.base_url = config.client.get("base_url", "http://127.0.0.1:8000/v1")
        self.timeout = int(config.client.get("timeout", 60))
        self.max_retries = int(config.client.get("max_retries", 3))

    def generate_completions(
        self, prompts: Union[str, List[str]], mode: str = "grpo_completions"
    ) -> List[str]:
        """Generate completions using vLLM server"""
        if isinstance(prompts, str):
            prompts = [prompts]

        # Generation config: support either nested per-mode dicts, or flat keys
        default_gen = {
            "max_tokens": int(self.config.generation.get("max_tokens", 512)),
            "temperature": float(self.config.generation.get("temperature", 0.8)),
            "top_p": float(self.config.generation.get("top_p", 0.9)),
            "frequency_penalty": float(
                self.config.generation.get("frequency_penalty", 0.0)
            ),
            "presence_penalty": float(
                self.config.generation.get("presence_penalty", 0.0)
            ),
            "stop": self.config.generation.get("stop", None),
        }
        generation_config = self.config.generation.get(mode, default_gen)

        completions = []

        for prompt in prompts:
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        (
                            f"{self.base_url}/v1/completions"
                            if not self.base_url.rstrip("/").endswith("/v1")
                            else f"{self.base_url}/completions"
                        ),
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
