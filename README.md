# Práctica 1 — Síntesis de voz

**Procesamiento del Habla · Grado en Ingeniería Informática · ETSI · Universidad de Huelva**

Este repositorio es el punto de partida de la Práctica 1. Contiene el cuaderno
de trabajo y las funciones de apoyo; el enunciado completo, con las tareas y la
rúbrica, está en el documento de la práctica publicado en Moodle.

El material funciona **igual en el servidor NIMROD y en vuestro propio
ordenador**. El código detecta dónde se ejecuta y ajusta rutas y dispositivo de
cómputo sin que tengáis que cambiar nada.

---

## Opción A — Trabajar en NIMROD (recomendada)

Es la opción por defecto: tiene GPU, los modelos ya están cacheados y no hay
nada que instalar.

1. Con la red de la UHU o la VPN, entrad en `http://nimrod:7081`.
2. Crear workspace con la plantilla `user-gpu`, poniendo vuestro número de equipo.
3. En JupyterLab, celda nueva:

```python
import shutil, pathlib
shutil.copytree('/data/datasets/material-p1', '/data/equipo/p1-tts',
                dirs_exist_ok=True)
enlace = pathlib.Path.home() / 'equipo'
if not enlace.exists():
    enlace.symlink_to('/data/equipo')
print(sorted(p.name for p in pathlib.Path('/data/equipo/p1-tts').iterdir()))
```

4. Refrescad el panel de ficheros y abrid `equipo` → `p1-tts` → `p1_plantilla.ipynb`.

Para entregar, copiad vuestro trabajo a este repositorio y haced *commit*
(ver «Qué se entrega», más abajo).

## Opción B — Trabajar en vuestro ordenador

Viable si tenéis máquina suficiente. Sin GPU todo funciona, solo que más
despacio: lo que más se nota es el reconocedor de referencia de la sesión 3,
que puede tardar varios minutos en transcribir las 20 frases.

**Requisitos mínimos razonables:** 16 GB de RAM, 15 GB libres en disco y Python
3.10 o superior. Con GPU NVIDIA, mejor.

```bash
git clone <url-de-vuestro-repositorio>
cd <carpeta>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python comprobar_entorno.py      # comprueba que todo está en su sitio

jupyter lab
```

Si tenéis GPU NVIDIA y queréis aprovecharla, instalad PyTorch con CUDA **antes**
del `requirements.txt`:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

En macOS con chip Apple, PyTorch usa la CPU para este material; es suficiente.

---

## Qué hay en el repositorio

| Ruta | Qué es |
|---|---|
| `p1_plantilla.ipynb` | Cuaderno de trabajo. Por aquí se empieza |
| `scripts/tts_util.py` | Rutas, detección de entorno, carga del corpus, `sin_ruido()` |
| `scripts/adaptadores.py` | Envoltorios de Piper y MMS-TTS, y la plantilla para el vuestro |
| `scripts/analisis.py` | RTF, análisis acústico con Praat, bootstrap |
| `scripts/evaluacion.py` | Inteligibilidad, MOS y consistencia de voz |
| `frases_ejemplo.tsv` | Formato del corpus, con seis frases de ejemplo |
| `informe/informe.md` | Esqueleto del informe |
| `comprobar_entorno.py` | Verifica que las dependencias están instaladas |
| `comprobar_entrega.py` | Verifica que la entrega está completa |

Los ficheros de `scripts/` están comentados y explican **por qué** cada medida se
toma como se toma. Leedlos: de eso se pregunta en la defensa individual.

---

## Qué se entrega

Se entrega **en este repositorio**, con *commits* a lo largo de la práctica (no
uno solo al final: el historial forma parte de la evaluación del trabajo en
equipo).

```
informe/informe.pdf        el informe, máximo 15 páginas sin anexos
p1_plantilla.ipynb         con todas las celdas ejecutadas
frases.tsv                 vuestro corpus de 20 frases
scripts/                   con vuestras modificaciones y normaliza()
resultados/                todos los .csv y .png
muestras/                  5 frases x 3 sistemas, con nombres claros
mos_clave.json             la clave del panel MOS
```

Antes de entregar, ejecutad:

```bash
python comprobar_entrega.py
```

### Qué NO se sube

El `.gitignore` ya excluye el audio generado, los modelos descargados y las
cachés. **No los forcéis**: un repositorio con 74 MB de modelos por cada voz no
se puede clonar con comodidad y GitHub acabará rechazándolo.

La excepción es `muestras/`, que sí se entrega: son las 15 muestras del panel
MOS y ocupan poco.

---

## Avisos que no son errores

- `pthread_setaffinity_np failed`: ONNX Runtime intenta repartir hilos entre más
  núcleos de los que tiene asignados. Por eso el material envuelve las llamadas
  en `sin_ruido()`.
- `Warning: unauthenticated requests to the HF Hub`: la descarga funciona, solo
  va más lenta.
- En NIMROD, «Startup scripts are still running»: lo provoca el gestor de la
  cola, que corre permanentemente.
- `groups: cannot find name for group ID 993`: inofensivo, solo en NIMROD.

## Dos cosas que conviene saber antes de empezar

- **La columna `leida` del corpus** decide lo que vais a medir. Si escribís
  «treinta de septiembre» y el reconocedor devuelve «30/09», el WER saldrá
  altísimo aunque la voz se haya entendido perfectamente. Decidid la convención
  del equipo, dejadla escrita en el informe, y usad `evaluacion.resumen_wer()` y
  `evaluacion.revisar()`.
- **Piper no es «el rápido»**: medido en NIMROD, MMS-TTS en GPU resultó unas
  trece veces más rápido. Lo que Piper es, es el que no consume GPU. Son dos
  virtudes distintas y conviene no confundirlas en el informe.
