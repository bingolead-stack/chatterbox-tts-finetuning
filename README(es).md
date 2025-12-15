# Chatterbox TTS Streaming

Chatterbox is an open source TTS model. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install chatterbox-streaming
```

## Training (lora.py)

### Dataset Setup
1. Place WAV files in `dataset/wavs/`
2. Create `dataset/metadata.csv` wiAquí tienes el **README traducido al español**, manteniendo la estructura y el contenido técnico intactos:

---

# Chatterbox TTS Streaming

Chatterbox es un modelo de **texto a voz (TTS)** de código abierto. Con licencia **MIT**, Chatterbox ha sido comparado con sistemas líderes de código cerrado como **ElevenLabs**, y es consistentemente preferido en evaluaciones lado a lado.

## Instalación

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install chatterbox-streaming
```

## Entrenamiento (lora.py)

### Preparación del Dataset

1. Coloca los archivos WAV en `dataset/wavs/`
2. Crea el archivo `dataset/metadata.csv` con el formato:

   ```
   filename.wav|texto de la transcripción
   ```
3. Duración del audio: **1–400 segundos** (los archivos fuera de este rango se omiten automáticamente)

### Preparación del Modelo

Coloca el archivo del modelo en la carpeta:

```
model/chatterbox-tts/
```

### Configuración

Edita la configuración en la parte superior de `lora.py`:

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

**Requisitos:**
GPU con CUDA y **18GB o más de VRAM**

### Iniciar el Entrenamiento

```bash
python lora.py
```

Las métricas de entrenamiento se guardan en `training_metrics.png` (se actualiza cada 2 segundos).

El modelo entrenado se puede encontrar en:

```
CHECKPOINT_DIR/merged_model
```

### Continuar el Entrenamiento desde un Modelo Previamente Entrenado

Para reanudar el entrenamiento desde un modelo ya entrenado:

1. Edita `lora.py` cerca de la línea ~40 y cambia:

```python
# Antes:
MODEL_DIR = "./model/chatterbox-tts/"

# Después:
MODEL_DIR = "./checkpoints_lora/merged_model/"
```

2. Inicia nuevamente el entrenamiento:

```bash
python lora.py
```

### Salidas del Entrenamiento

Se guardan en `checkpoints_lora/`:

* `checkpoint_epoch{N}_step{M}.pt` – Checkpoints de entrenamiento
* `final_lora_adapter.pt` – Pesos del adaptador LoRA
* `merged_model/` – Modelo completo fusionado (listo para usar)

## Pruebas (main.py)

### Uso Básico

Edita `main.py` para definir tu texto y un audio de referencia opcional:

```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar el modelo entrenado
model = ChatterboxTTS.from_local(
    ckpt_dir='checkpoints_lora/merged_model',
    device=device
)

# Texto a sintetizar
text = "Tu texto aquí"

# Opcional: usar audio de referencia para clonación de voz
AUDIO_PROMPT_PATH = "sample.wav"
model.prepare_conditionals(AUDIO_PROMPT_PATH)

# Generar y guardar audio
wav = model.generate(text)
ta.save("out.wav", wav, model.sr)
```

### Ejecutar la Prueba

```bash
python main.py
```

El archivo de salida se guarda como:

```
out.wav
```

---

Si quieres, también puedo:

* Localizarlo para **español latino o español de España**
* Adaptarlo para **GitHub README con badges**
* Traducir comentarios del código
* Simplificar el README para usuarios no técnicos
th format: `filename.wav|transcription text`
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
