"""Weights & Biases logging utilities."""

import os
import wandb
from datetime import datetime
from typing import Dict, Any, Optional

class WandBLogger:
    """A lightweight Weights & Biases logger with offline support."""

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        offline: bool = True,
        disable: bool = False,
    ):
        """
        Initialize the W&B logger.

        Args:
            project: The name of the W&B project.
            entity: The W&B entity (user or team).
            name: The name of the run.
            config: A dictionary of hyperparameters.
            offline: If True, run in offline mode.
            disable: If True, disable all W&B logging.
        """
        self.disable = disable or os.environ.get("WANDB_DISABLED", "false").lower() == "true"
        if self.disable:
            print("W&B logging is disabled.")
            return

        # Set a timeout for wandb.init() to avoid long waits
        os.environ["WANDB_INIT_TIMEOUT"] = "10"

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = name if name else f"run-{timestamp}"
        project_name = f"{project}-{timestamp}"

        mode = "offline" if offline else "online"

        try:
            wandb.init(
                project=project_name,
                entity=entity,
                name=run_name,
                config=config,
                reinit=True,
                mode=mode,
            )
            print(f"W&B: Initialized project '{project_name}' in {mode} mode.")
        except Exception as e:
            print(f"W&B: Failed to initialize. Disabling for this run. Error: {e}")
            self.disable = True


    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """
        Log data to W&B.

        Args:
            data: A dictionary of data to log.
            step: The current step or epoch.
        """
        if not self.disable:
            try:
                wandb.log(data, step=step)
            except Exception as e:
                print(f"W&B: Failed to log data. Error: {e}")


    def finish(self):
        """Finish the W&B run."""
        if not self.disable:
            try:
                wandb.finish()
                print("W&B: Finished run.")
            except Exception as e:
                print(f"W&B: Failed to finish run. Error: {e}")
