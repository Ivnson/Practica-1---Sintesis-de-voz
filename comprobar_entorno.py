#!/usr/bin/env python3
"""
Comprueba que el entorno esta listo para la Practica 1.

Uso:   python comprobar_entorno.py

Funciona tanto en NIMROD como en un ordenador propio. No descarga modelos:
solo verifica que las piezas estan instaladas y que el computo funciona.
"""

import importlib
import pathlib
import sys

OK, FALLO, AVISO = "[ OK ]", "[FALLO]", "[AVISO]"
fallos = 0


def comprueba(nombre, paquete=None, atributo="__version__"):
    global fallos
    try:
        mod = importlib.import_module(paquete or nombre)
        print(f"{OK} {nombre:22s} {getattr(mod, atributo, '?')}")
        return mod
    except Exception as exc:
        fallos += 1
        print(f"{FALLO} {nombre:22s} {type(exc).__name__}: {exc}")
        return None


print("=" * 62)
print("COMPROBACION DEL ENTORNO - Practica 1")
print("=" * 62)

print(f"\nPython {sys.version.split()[0]}")
if sys.version_info < (3, 10):
    print(f"{AVISO} Se recomienda Python 3.10 o superior")

print("\n--- Nucleo ---")
torch = comprueba("torch")
comprueba("torchaudio")

if torch is not None:
    print("\n--- Computo ---")
    if torch.cuda.is_available():
        print(f"{OK} GPU: {torch.cuda.get_device_name(0)}")
        try:
            x = torch.randn(512, 512, device="cuda")
            _ = (x @ x).sum().item()
            print(f"{OK} operacion en GPU correcta")
        except Exception as exc:
            fallos += 1
            print(f"{FALLO} la GPU se detecta pero falla: {exc}")
    else:
        print(f"{AVISO} Sin GPU: todo funcionara en CPU, mas despacio.")
        print("       Lo que mas se nota es el reconocedor de la sesion 3.")

print("\n--- Sintesis ---")
comprueba("piper", "piper")
comprueba("transformers")

print("\n--- Evaluacion ---")
comprueba("faster_whisper")
comprueba("jiwer")
comprueba("speechbrain")

print("\n--- Audio y fonetica ---")
comprueba("librosa")
comprueba("soundfile")
comprueba("parselmouth", "parselmouth", "VERSION")

print("\n--- Datos y figuras ---")
comprueba("numpy")
comprueba("pandas")
comprueba("matplotlib")

print("\n--- Material de la practica ---")
raiz = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(raiz / "scripts"))
try:
    import tts_util

    print(f"{OK} entorno detectado: {tts_util.ENTORNO}")
    print(f"{OK} carpeta de trabajo: {tts_util.BASE}")
except Exception as exc:
    fallos += 1
    print(f"{FALLO} no se pudo importar scripts/tts_util.py: {exc}")

for fichero in ("p1_plantilla.ipynb", "frases_ejemplo.tsv",
                "scripts/adaptadores.py", "scripts/analisis.py",
                "scripts/evaluacion.py"):
    if (raiz / fichero).exists():
        print(f"{OK} {fichero}")
    else:
        fallos += 1
        print(f"{FALLO} falta {fichero}")

print("\n" + "=" * 62)
if fallos:
    print(f"RESULTADO: {fallos} problema(s). Revisad lo marcado como FALLO.")
    print("Si falta algun paquete:  pip install -r requirements.txt")
    sys.exit(1)
print("RESULTADO: entorno listo.")
sys.exit(0)
