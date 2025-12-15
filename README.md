# Chatterbox TTS Streaming

Chatterbox is an open source TTS model. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install .
```

## Training (lora.py)

### Dataset Setup
1. Place WAV files in `dataset/wavs/`
2. Create `dataset/metadata.csv` with format: `filename.wav|transcription text`
3. Audio duration: 1-400 seconds (auto-skipped if outside range)

### Model Setup
Place model file to `model/chatterbox-tts/` folder

### Configuration
Edit config at top of `lora.py`:
```python
WAVS_DIR = "./dataset/wavs"
METADATA_FILE = "./dataset/metadata.csv"
MODEL_DIR = "./model/chatterbox-tts/"
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 2e-5
LORA_RANK = 32
LORA_ALPHA = 64
CHECKPOINT_DIR = "checkpoints_lora"
```

**Requirements:** CUDA GPU with 18GB+ VRAM

### Start Training
```bash
python lora.py
```

Training metrics saved to `training_metrics.png` (updates every 2s).
You can find trained model from CHECKPOINT_DIR\merged_model

### Continue Training from Previous trained Model

To resume training from a previously trained model:

1. Edit `lora.py` line ~40, change:
```python
# From:
MODEL_DIR = "./model/chatterbox-tts/"

# To:
MODEL_DIR = "./checkpoints_lora/merged_model/"
```


2. Start training
```python
python lora.py
```

### Training Outputs
Saved in `checkpoints_lora/`:
- `checkpoint_epoch{N}_step{M}.pt` - Training checkpoints
- `final_lora_adapter.pt` - LoRA adapter weights
- `merged_model/` - Complete merged model (ready to use)

## Testing (main.py)

### Basic Usage
Edit `main.py` to set your text and optional audio prompt:

```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model
model = ChatterboxTTS.from_local(ckpt_dir='checkpoints_lora/merged_model', device=device)

# Set text to synthesize
text = "Your text here"

# Optional: Use reference audio for voice cloning
AUDIO_PROMPT_PATH = "sample.wav"
model.prepare_conditionals(AUDIO_PROMPT_PATH)

# Generate and save
wav = model.generate(text)
ta.save("out.wav", wav, model.sr)
```

### Run Test
```bash
python main.py
```

Output saved to `out.wav`
