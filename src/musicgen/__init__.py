import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID = "facebook/musicgen-small"
DEVICE = "cpu"

PROMPT="Latin Afrobeat fusion, tropical pop style. Smooth mid-tempo rhythm, bouncing West African percussion mixed with a subtle reggaeton dembow bounce, warm acoustic guitar riffs, and dreamy synth plucks. Airy, melodic, and high-pitched English male vocals with a sweet, romantic delivery. Catchy vocal hooks, sunny beach vibe, smooth urban radio production"

print("Cargando Musicgen")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model = model.to(DEVICE)
model.eval()

print("Preparando prompt...")

inputs = processor(text=[PROMPT],padding=True,return_tensors="pt").to(DEVICE)

print("Generando musica....")
with torch.inference_mode():
    audio = model.generate(
    **inputs,do_sample=True, guidance_scale=3.0, max_new_tokens=2000 # 5segundos- 750 15 seg    
    )
    sample_rate = model.config.audio_encoder.sampling_rate
    waveform = audio[0,0].cpu().float().numpy()

    write ("musica.wav",sample_rate, waveform)
    print("musica generada")

    
    
