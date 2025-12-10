import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ChatterboxTTS.from_local(ckpt_dir='checkpoints_lora\merged_model', device=device)
# ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]

text = "en efecto parecia una torrecilla gotica, estas curvas del gusto sobre todo del cuello, a la marquesa se le antojaba un caballo de ajedrez por lo demás a ella y a sus tres hermanas la llamaban los plebeyos las tres desgracias"

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH="sample.wav"
model.prepare_conditionals(AUDIO_PROMPT_PATH)

wav = model.generate(text)
ta.save(f"out.wav", wav, model.sr)