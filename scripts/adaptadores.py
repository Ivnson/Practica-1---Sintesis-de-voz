"""
Adaptadores de sistemas TTS — Practica 1.

Todos los sistemas se envuelven en una funcion con la MISMA firma:

    sintetiza(texto: str, ruta: Path | str, **parametros) -> None

de modo que el resto del codigo (medidas, prosodia, evaluacion) funcione
igual sea cual sea el sistema.
"""

import pathlib
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForTextToWaveform

from tts_util import MODELOS, dispositivo, sin_ruido
import os

# Variables de entorno clave para que PyTorch intente derivar operaciones a CPU
# si la gráfica de Mac (MPS) no soporta una instrucción específica del modelo.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["SUNO_ENABLE_MPS_FALLBACK"] = "True"


# ===========================================================================
# Candidato A — Piper
# ===========================================================================
def descargar_voz_piper(nombre="es_ES-sharvard-medium"):
    import subprocess
    import sys

    destino = MODELOS / "piper"
    destino.mkdir(parents=True, exist_ok=True)
    onnx = destino / f"{nombre}.onnx"
    if onnx.exists():
        print(f"{nombre} ya descargada ({onnx.stat().st_size / 1e6:.0f} MB)")
        return onnx

    print(f"Descargando {nombre} ...")
    subprocess.run(
        [sys.executable, "-m", "piper.download_voices", nombre, "--data-dir", str(destino)],
        check=True,
    )
    print(f"Listo: {onnx.stat().st_size / 1e6:.0f} MB")
    return onnx


def crear_piper(nombre="es_ES-sharvard-medium", usar_gpu=False):
    import wave
    from piper import PiperVoice, SynthesisConfig

    onnx = descargar_voz_piper(nombre)
    with sin_ruido():
        voz = PiperVoice.load(str(onnx), use_cuda=usar_gpu)

    def sintetiza(texto, ruta, **parametros):
        cfg = SynthesisConfig(**parametros) if parametros else None
        ruta = pathlib.Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with sin_ruido(), wave.open(str(ruta), "wb") as f:
            voz.synthesize_wav(texto, f, syn_config=cfg)

    sintetiza.nombre = f"piper_{nombre}"
    sintetiza.admite_parametros = True
    return sintetiza


# ===========================================================================
# Candidato B — MMS-TTS de Meta
# ===========================================================================
def crear_mms(modelo="facebook/mms-tts-spa", device=None):
    import soundfile as sf
    import torch
    from transformers import AutoTokenizer, VitsModel

    # Detección universal de hardware para ignorar la limitación de tts_util
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
    print(f"Cargando MMS en: {device}")
    red = VitsModel.from_pretrained(modelo).to(device)
    red.eval()
    tok = AutoTokenizer.from_pretrained(modelo)

    def sintetiza(texto, ruta, **parametros):
        ruta = pathlib.Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        x = tok(texto, return_tensors="pt").to(device)
        with torch.no_grad():
            y = red(**x).waveform.squeeze().cpu().numpy()
        sf.write(str(ruta), y, red.config.sampling_rate)

    sintetiza.nombre = modelo.split("/")[-1]
    sintetiza.admite_parametros = False
    return sintetiza


# ===========================================================================
# Registro de candidatos
# ===========================================================================
def registrar(candidatos, nombre, constructor, *args, **kwargs):
    try:
        candidatos[nombre] = constructor(*args, **kwargs)
        print(f"[ OK ] {nombre}")
    except Exception as exc:
        print(f"[FALLO] {nombre}")
        print(f"        {type(exc).__name__}: {exc}")
        print("        Anotad este error en el informe y descartad el candidato.")
    return candidatos


# ===========================================================================
# Candidato C — Bark (Suno)
# ===========================================================================
def crear_bark(usar_gpu=True):
    # Detección universal de hardware para que soporte tanto Nvidia como Mac
    if usar_gpu:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"  # IMPORTANTE: mps a secas, sin el ":0"
        else:
            device = "cpu"
    else:
        device = "cpu"

    print(f"Usando Bark en {device} (usar_gpu={usar_gpu})")
    
    modelo_id = "suno/bark-small"
    processor = AutoProcessor.from_pretrained(modelo_id)
    model = AutoModelForTextToWaveform.from_pretrained(modelo_id).to(device)
    
    sampling_rate = model.generation_config.sample_rate

    def sintetizar(texto, ruta_salida, **kwargs):
        inputs = processor(texto, voice_preset="v2/es_speaker_0")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            audio_array = model.generate(**inputs)
            
        audio = audio_array.squeeze().cpu().numpy()
        sf.write(str(ruta_salida), audio, sampling_rate)

    sintetizar.admite_parametros = False
    sintetizar.nombre_modelo = modelo_id
    sintetizar.device = device

    return sintetizar