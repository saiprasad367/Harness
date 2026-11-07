"""
Turnkey script to run Kyutai Moshi full-duplex server on a free Kaggle GPU (T4/P100).
Exposes a public secure WebSocket URL using Kyutai's built-in Gradio tunnel.

Instructions for Kaggle:
1. Open https://www.kaggle.com and create a New Notebook.
2. In Notebook Settings (right panel):
   - Accelerator: GPU T4 x2 or GPU P100 (Free 30 hrs/week)
   - Internet: ON (required to download weights & tunnel)
3. Paste and run this script.
4. Copy the generated public URL (e.g., https://xxxx.gradio.live).
5. Paste it into the Harness Dashboard (http://localhost:8501) -> Tab 1 -> Connect to Moshi.
"""
import os
import sys
import subprocess

print("=" * 70)
print("🎙️ KYUTAI MOSHI — KAGGLE GPU FULL-DUPLEX BACKEND SERVER")
print("=" * 70)

# Step 1: Install Kyutai Moshi and dependencies
print("\n[1/3] Installing Kyutai Moshi and dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "moshi", "gradio"], check=True)

# Step 2: Verify GPU Availability
import torch
if not torch.cuda.is_available():
    print("⚠️ WARNING: GPU is not detected! In Kaggle, set Accelerator to 'GPU T4' in notebook settings.")
else:
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✅ GPU Detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")

# Step 3: Launch Genuine Moshi Server with Public Gradio Tunnel
print("\n[2/3] Launching Genuine Kyutai Moshi Server with --gradio-tunnel...")
print("Downloading Kyutai weights (kyutai/moshiko-pytorch-bf16) and generating public endpoint...")
print("=" * 70)
print("👉 Look for the 'Running on public URL: https://xxxx.gradio.live' link below.")
print("👉 Copy that link and paste it into your local Harness dashboard at http://localhost:8501")
print("=" * 70 + "\n")

# Execute official Kyutai Moshi server
cmd = [
    sys.executable,
    "-m",
    "moshi.server",
    "--gradio-tunnel",
    "--hf-repo",
    "kyutai/moshiko-pytorch-bf16",
]

try:
    subprocess.run(cmd)
except KeyboardInterrupt:
    print("\nMoshi server terminated by user.")

