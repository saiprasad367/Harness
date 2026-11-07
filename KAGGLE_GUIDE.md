# 🚀 Step-by-Step Guide: Running Kyutai Moshi on Free Kaggle GPU

This guide explains step-by-step how to run the genuine **Kyutai Moshi** 7B full-duplex speech model on **Kaggle's free GPU instances (T4 x2 or P100)** and connect it to your local Adaptive Voice Agent Harness.

---

## 📋 Prerequisites
1. A free account on [kaggle.com](https://www.kaggle.com).
2. **Kaggle Phone Verification**: Kaggle requires a one-time SMS verification to enable internet access in notebooks.
   - Check if your account is verified: Go to [kaggle.com/settings](https://www.kaggle.com/settings) &rarr; scroll to **Phone verification** &rarr; Verify your mobile number if not already done.

---

## 🛠️ Step-by-Step Instructions

### Step 1: Create a New Kaggle Notebook
1. Go to [kaggle.com](https://www.kaggle.com).
2. Click the **"+ Create"** button in the top-left sidebar &rarr; select **"New Notebook"**.

---

### Step 2: Configure Free GPU & Internet (Crucial!)
Look at the **Notebook options** panel on the right side of the screen:
1. **Accelerator**: Click on it and select **`GPU T4 x2`** (or **`GPU P100`**).
2. **Internet**: Toggle the switch to **`ON`** (required to download model weights and generate the tunnel).

> 💡 *Note: Kaggle provides 30 hours of free GPU per week for every verified user.*

---

### Step 3: Add the Notebook Code
You have two easy options:

#### Option A: Upload the File (Fastest)
1. In the Kaggle notebook menu, click **File** &rarr; **Upload notebook**.
2. Select [`kaggle_moshi_server.ipynb`](file:///C:/Users/saipr/Videos/Harness/kaggle_moshi_server.ipynb) from this repository.

#### Option B: Copy & Paste Code Cells
If you prefer pasting into a blank notebook:

**Cell 1:**
```python
# Check GPU
!nvidia-smi
```

**Cell 2:**
```python
# Install Kyutai Moshi and Gradio
!pip install -q -U moshi gradio
```

**Cell 3 (Memory-Optimized for 15GB Tesla T4):**
```python
# 1. Configure PyTorch execution environment
os.environ["NO_CUDA_GRAPH"] = "1"
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Balance transformer across GPU 0 & GPU 1
import moshi.modules.transformer as tf_mod
orig_tf_forward = tf_mod.StreamingTransformer.forward
def multi_gpu_tf_forward(self, x, *args, **kwargs):
    target_device = x.device
    for layer in self.layers:
        layer_dev = next(layer.parameters()).device
        if x.device != layer_dev:
            x = x.to(layer_dev)
        x = layer(x, *args, **kwargs)
    return x.to(target_device)
tf_mod.StreamingTransformer.forward = multi_gpu_tf_forward

import moshi.models.loaders as loaders

class CrossDeviceMimi:
    def __init__(self, mimi, mimi_device, target_device="cuda:0"):
        self._mimi = mimi
        self._mimi_device = torch.device(mimi_device)
        self._target_device = torch.device(target_device)

    def encode(self, chunk):
        chunk = chunk.to(self._mimi_device)
        codes = self._mimi.encode(chunk)
        return codes.to(self._target_device)

    def decode(self, tokens):
        tokens = tokens.to(self._mimi_device)
        pcm = self._mimi.decode(tokens)
        return pcm.to(self._target_device)

    def __getattr__(self, name):
        return getattr(self._mimi, name)

orig_get_mimi = loaders.CheckpointInfo.get_mimi
def multi_gpu_get_mimi(self, device="cuda"):
    mimi_dev = "cuda:1" if torch.cuda.device_count() > 1 else "cpu"
    raw_mimi = orig_get_mimi(self, device=mimi_dev)
    print(f"✅ Mimi loaded on {mimi_dev}!")
    return CrossDeviceMimi(raw_mimi, mimi_device=mimi_dev, target_device=device)

loaders.CheckpointInfo.get_mimi = multi_gpu_get_mimi

orig_load_file = loaders.load_file
def safe_load_file(filename, device="cuda"):
    cpu_state = orig_load_file(filename, device="cpu")
    gc.collect()
    torch.cuda.empty_cache()
    has_two_gpus = torch.cuda.device_count() > 1
    target_state = {}
    for k, v in cpu_state.items():
        target_dev = str(device)
        if has_two_gpus and k.startswith("transformer.layers."):
            try:
                layer_idx = int(k.split(".")[2])
                if layer_idx >= 16:
                    target_dev = "cuda:1"
            except Exception:
                pass
        if v.dtype.is_floating_point:
            if k.startswith('condition_provider.') or k.startswith('fuser.'):
                v = v.float()
            else:
                v = v.to(torch.bfloat16)
        target_state[k] = v.to(target_dev)
    del cpu_state
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Model balanced: Layers 0-15 on GPU 0, Layers 16-31 on GPU 1!")
    return target_state

loaders.load_file = safe_load_file

# Set CLI arguments BEFORE importing moshi.server (server auto-executes main() on import)
sys.argv = [
    "server.py",
    "--gradio-tunnel",
    "--hf-repo", "kyutai/moshiko-pytorch-bf16"
]
import moshi.server
```

---

### Step 4: Run the Notebook & Wait for Model Loading
1. Click **"Run All"** (or press `Shift + Enter` on each cell sequentially).
2. **Cell 1** will confirm your GPU (`Tesla T4` with ~15-16 GB VRAM).
3. **Cell 2** will install `moshi` (~30-60 seconds).
4. **Cell 3** will start downloading the official weights (`kyutai/moshiko-pytorch-bf16`, ~15 GB) from Hugging Face:
   - *Download time*: Usually takes 2 to 4 minutes on Kaggle's high-speed cloud connection.
   - Once ready, the cell will output:
     ```text
     Running on local URL:  http://127.0.0.1:8998
     Running on public URL: https://xxxxxxxxxxxx.gradio.live
     ```

---

### Step 5: Connect Your Local Harness
1. Copy the public link generated by Kaggle:
   ```text
   https://xxxxxxxxxxxx.gradio.live
   ```
2. Start your local Streamlit Harness (if not already running):
   ```powershell
   streamlit run app.py
   ```
3. Open your browser at `http://localhost:8501`.
4. In **Tab 1: Live Voice Interaction Studio**:
   - Paste your URL into the **Kyutai Moshi Endpoint** field.
   - Click **"🔗 Connect to Moshi"**.
5. The badge will instantly change from `DISCONNECTED / OFFLINE` to **`CONNECTED`** (with green pill and latency metrics).
6. Click **"🎙️ Start Live Session"** and speak into your microphone!

---

## ⚡ Automated 1-Click Launch via Kaggle API (Alternative)

If you don't want to use the browser interface, you can deploy directly from your local terminal or from the Streamlit UI:

1. Download your API key from [kaggle.com/settings](https://www.kaggle.com/settings) &rarr; **Create New Token** (`kaggle.json`).
2. In the Harness dashboard (**Tab 1**), open the **⚡ Kaggle Free GPU Hub** expander.
3. Paste your **Kaggle Username** and **API Key**.
4. Click **"🚀 Deploy to Kaggle GPU"**.
5. The harness automatically uploads the notebook, triggers the GPU session, monitors output logs, and grabs the Gradio URL for you.

---

## ❓ Frequently Asked Questions & Troubleshooting

### Q1: The Internet toggle is grayed out in Kaggle.
- **Cause**: Kaggle requires phone verification to enable internet access.
- **Fix**: Go to [kaggle.com/settings](https://www.kaggle.com/settings), verify your phone number via SMS, then refresh the notebook.

### Q2: CUDA out of memory error.
- **Cause**: CPU or default session was selected instead of GPU.
- **Fix**: Open the right sidebar &rarr; **Notebook options** &rarr; select **`GPU T4 x2`**.

### Q3: How long does the Kaggle session stay active?
- An interactive Kaggle notebook stays running for up to **12 hours** as long as you keep the browser tab open. If you close the browser tab, the idle timeout will shut it down after ~40-60 minutes.

### Q4: Is the Gradio tunnel URL secure?
- Yes. The Gradio tunnel uses encrypted `wss://` (WebSocket Secure) with SSL certificates, which your local Harness handles automatically.

