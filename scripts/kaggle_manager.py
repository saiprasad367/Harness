"""
Automated Kaggle GPU Deployment Manager for Kyutai Moshi.
Uses Kaggle API to push and run a GPU notebook (T4 x2) on Kaggle with Gradio tunneling,
and monitor output logs to retrieve the public WebSocket URL.
"""
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import dotenv
    dotenv.load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = ROOT_DIR / "config" / "kaggle_credentials.json"
KERNEL_DIR = ROOT_DIR / "outputs" / "kaggle_kernel"
KAGGLE_CFG_DIR = ROOT_DIR / "config" / "kaggle"


def _patch_kaggle_sandbox_safety():
    """Patches Kaggle SDK to prevent Windows sandbox PermissionError on ~/.kaggle."""
    try:
        import kagglesdk.kaggle_env
        kagglesdk.kaggle_env._get_access_token_from_file = lambda p: None
    except Exception:
        pass


class KaggleMoshiManager:
    """Manages remote Moshi execution on free Kaggle GPU instances."""

    def __init__(self, username: Optional[str] = None, api_key: Optional[str] = None):
        self.username = username
        self.api_key = api_key
        self._load_saved_credentials()

    def _load_saved_credentials(self):
        """Loads cached credentials from config file or environment if not provided."""
        if not self.username or not self.api_key:
            if CREDENTIALS_FILE.exists():
                try:
                    with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.username = self.username or data.get("username")
                        self.api_key = self.api_key or data.get("key")
                except Exception as e:
                    log.warning(f"Failed to read {CREDENTIALS_FILE}: {e}")

        self.username = self.username or os.environ.get("KAGGLE_USERNAME")
        self.api_key = self.api_key or os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN")

    def save_credentials(self, username: str, api_key: str):
        """Saves credentials securely to local config folder."""
        self.username = username.strip()
        self.api_key = api_key.strip()
        CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CREDENTIALS_FILE, "w", encoding="utf-8") as f:
            json.dump({"username": self.username, "key": self.api_key}, f, indent=2)
        os.environ["KAGGLE_USERNAME"] = self.username
        os.environ["KAGGLE_KEY"] = self.api_key
        if self.api_key.startswith("KGAT_"):
            os.environ["KAGGLE_API_TOKEN"] = self.api_key

    def _setup_api_client(self):
        """Prepares KaggleApi client with safe configuration."""
        _patch_kaggle_sandbox_safety()
        KAGGLE_CFG_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["KAGGLE_CONFIG_DIR"] = str(KAGGLE_CFG_DIR)

        if self.username:
            os.environ["KAGGLE_USERNAME"] = self.username
        if self.api_key:
            os.environ["KAGGLE_KEY"] = self.api_key
            if self.api_key.startswith("KGAT_"):
                os.environ["KAGGLE_API_TOKEN"] = self.api_key

        with open(KAGGLE_CFG_DIR / "kaggle.json", "w", encoding="utf-8") as f:
            json.dump({"username": self.username or "", "key": self.api_key or ""}, f)

        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api

    def verify_credentials(self) -> Tuple[bool, str]:
        """Validates credentials against Kaggle API."""
        if not self.username or not self.api_key:
            return False, "Kaggle username and API key are required."

        try:
            api = self._setup_api_client()
            return True, f"Authenticated successfully as '{self.username}'."
        except Exception as e:
            return False, f"Kaggle authentication failed: {str(e)}"

    def prepare_kernel_package(self) -> Path:
        """Prepares kernel directory with notebook and kernel-metadata.json."""
        if not self.username:
            raise ValueError("Kaggle username must be set before preparing kernel.")

        KERNEL_DIR.mkdir(parents=True, exist_ok=True)

        # Source notebook
        src_nb = ROOT_DIR / "kaggle_moshi_server.ipynb"
        dest_nb = KERNEL_DIR / "kaggle_moshi_server.ipynb"
        shutil.copy(src_nb, dest_nb)

        # Kernel metadata
        metadata = {
            "id": f"{self.username}/kyutai-moshi-gpu-server",
            "title": "Kyutai Moshi GPU Server",
            "code_file": "kaggle_moshi_server.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }

        with open(KERNEL_DIR / "kernel-metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return KERNEL_DIR

    def push_kernel(self) -> Tuple[bool, str]:
        """Pushes the Moshi server notebook to Kaggle."""
        ok, msg = self.verify_credentials()
        if not ok:
            return False, msg

        self.prepare_kernel_package()

        try:
            api = self._setup_api_client()
            response = api.kernels_push(str(KERNEL_DIR), acc="NvidiaTeslaT4")
            kernel_slug = f"{self.username}/kyutai-moshi-gpu-server"
            return True, f"Kernel pushed successfully: {kernel_slug}. Response: {response}"
        except Exception as e:
            return False, f"Failed to push kernel to Kaggle: {str(e)}"

    def check_status(self) -> Dict[str, any]:
        """Checks the execution status of the pushed kernel."""
        ok, msg = self.verify_credentials()
        if not ok:
            return {"status": "error", "message": msg}

        try:
            api = self._setup_api_client()
            kernel_slug = f"{self.username}/kyutai-moshi-gpu-server"
            status_obj = api.kernels_status(kernel_slug)
            return {
                "status": getattr(status_obj, "status", str(status_obj)),
                "kernel": kernel_slug,
                "message": getattr(status_obj, "failure_message", None),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def extract_gradio_url_from_logs(self) -> Optional[str]:
        """Fetches output logs of the kernel and searches for Gradio live URL."""
        ok, _ = self.verify_credentials()
        if not ok:
            return None

        try:
            api = self._setup_api_client()
            kernel_slug = f"{self.username}/kyutai-moshi-gpu-server"
            download_dir = ROOT_DIR / "outputs" / "kaggle_log"
            download_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                api.kernels_output(kernel_slug, str(download_dir))
            except Exception as dl_err:
                log.debug(f"kernels_output: {dl_err}")

            full_text = ""
            for log_file in download_dir.glob("*"):
                if log_file.is_file():
                    try:
                        content = log_file.read_text(encoding="utf-8", errors="ignore")
                        full_text += "\n" + content
                    except Exception:
                        pass

            # Match Gradio live URL pattern: https://xxxx.gradio.live
            match = re.search(r"https://[a-zA-Z0-9-]+\.gradio\.live", full_text)
            if match:
                return match.group(0)
            return None
        except Exception as e:
            log.warning(f"Error checking kernel output logs: {e}")
            return None

