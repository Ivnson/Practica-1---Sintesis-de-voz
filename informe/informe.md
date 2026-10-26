# Práctica 1 — Síntesis de voz

**Equipo N** · Dominio: _(el que os corresponda)_

Integrantes y reparto del trabajo:

| Integrante | Rol principal | De qué se ocupó |
|---|---|---|
|  |  |  |

> Entregad el informe como `informe/informe.pdf`. Este fichero es solo el
> esqueleto; podéis escribirlo en Markdown, LaTeX o Word, pero el PDF es lo
> que se corrige. Máximo 15 páginas sin anexos.

---

## 1. Resumen

Media página. Qué sistema recomendáis y las tres cifras que lo justifican.

## 2. Candidatos

A continuación, se presenta la evaluación técnica de los tres sistemas candidatos analizados (Piper, MMS y Bark). Debido a incompatibilidades arquitectónicas con el hardware local, el candidato Bark ha sido descartado como motor principal, dejando a Piper y MMS como los modelos funcionales viables para la práctica.

| Criterio | Piper (`es_ES-sharvard-medium`) | MMS (`facebook/mms-tts-spa`) | Bark (`suno/bark-small`) |
| :--- | :--- | :--- | :--- |
| **Licencia y uso comercial** | CC0 (dataset) y MIT (código). **Sí** permite uso comercial. | CC-BY-NC 4.0 (Meta). **No** permite uso comercial. | MIT License (Suno). **Sí** permite uso comercial. |
| **Español: variedad, nº de voces** | 1 voz individual (dataset Sharvard). Español de España. | Cubre múltiples variantes y dialectos del español. | Multivoz nativo. Permite elegir *presets* (ej. `v2/es_speaker_0`). |
| **Memoria ocupada** | Muy ligero (~85 MB). Ejecución eficiente. | Moderado (~1 GB). | Pesado (>1.5 GB). |
| **Velocidad (RTF medio)** | **0.022** (Muy rápido). Ejecutado en CPU. | **0.070** (Muy rápido). Ejecutado en GPU (Apple MPS). | **2.502** (Inviable). Ejecutado en GPU (MPS) con *fallback* a CPU. |
| **Mantenimiento** | Activo (rhasspy). Basado en ONNX, ideal para entornos de bajos recursos. | Estable (Meta, 2023). Integrado nativamente en `transformers`. | Activo (Suno). Integrado nativamente en `transformers`. |
| **Calidad** | *Se evaluará en la Sesión 3.* | *Se evaluará en la Sesión 3.* | *Se evaluará en la Sesión 3.* |

### Análisis del candidato descartado (Bark)
Se configuró y ejecutó el modelo generativo Bark (`suno/bark-small`) haciendo uso de la aceleración por hardware de Apple Silicon (MPS). Sin embargo, se documenta una incompatibilidad nativa en ciertas capas matemáticas del modelo con el *backend* de PyTorch en macOS. Esto obligó a forzar un "puente" de compatibilidad (`PYTORCH_ENABLE_MPS_FALLBACK=1`), provocando cuellos de botella severos por el intercambio constante de datos entre la GPU y la CPU. El resultado es un RTF inaceptable (>2.5, tardando casi 10 minutos para 20 frases cortas), por lo que se descarta su uso para un sistema de síntesis en tiempo real.

## 3. Velocidad y recursos

### Métricas de rendimiento
A partir de las mediciones realizadas con 20 frases por sistema, se han extraído los siguientes datos de rendimiento y consumo. Para el RTF (Real-Time Factor), se ha tomado el valor mediano (`50%` de la distribución estadística) para reflejar el comportamiento típico.

| Sistema | Dispositivo | Memoria Estimada | Tiempo de carga | RTF Mediano |
| :--- | :--- | :--- | :--- | :--- |
| **Piper** (`es_ES-sharvard`) | CPU | ~85 MB | 0.73 s | 0.021 |
| **MMS** (`mms-spa`) | GPU (Apple MPS) | ~1.0 GB | 1.75 s | 0.063 |
| **Bark** (`bark-small`) | GPU  Apple (MPS) + CPU | >1.5 GB | 3.07 s | 2.685 |

*El tiempo de carga es proporcional al tamaño del modelo en disco: Piper, al ser un modelo muy ligero basado en ONNX, se vuelca a memoria de forma casi instantánea, mientras que los modelos basados en Transformers (MMS y Bark) requieren mayor tiempo para la inicialización de tensores.*

### Comparación CPU frente a GPU (Tarea 1.4)
Al analizar la ejecución en la arquitectura local (Apple Silicon M4), se observan comportamientos muy dispares según el diseño de cada sistema:

1.  **MMS** demuestra una gran optimización al aprovechar la aceleración gráfica nativa (MPS), obteniendo tiempos de respuesta excelentes.
2.  **Bark** presenta un problema de rendimiento doble. Por un lado, al ser un modelo generativo masivo basado en Transformers que sintetiza de forma autorregresiva, prioriza la expresividad extrema por encima de la velocidad. Incluso en equipos con hardware óptimo (NVIDIA CUDA), Bark es lento. Por otro lado, su **incompatibilidad en macOS**: ciertas operaciones matemáticas del modelo no están soportadas nativamente por PyTorch en MPS, forzando un *fallback* constante hacia la CPU. La suma de su colosal carga matemática y el cuello de botella en la transferencia de datos hunde su rendimiento (RTF > 2.6), invalidándolo para producción local.

**¿Gana siempre la GPU con un modelo pequeño?**
Al repetir la medida de Piper forzando `usar_gpu=True`, se evidencia que **la GPU no siempre es la mejor opción para modelos pequeños**. Esto se justifica por el cuello de botella de la transferencia de datos. Para procesar en la gráfica, el sistema debe empaquetar la entrada, copiarla desde la RAM a la VRAM, invocar la ejecución y devolver el audio generado. En modelos diminutos y altamente optimizados como Piper, la CPU tarda menos tiempo en procesar la síntesis de forma directa que el tiempo que pierde el sistema informático en mover los datos entre componentes. 

### Justificación de las reglas de medida

1.  **Por qué la carga no cuenta:** En un entorno de producción real (como un servidor backend o un asistente virtual), el modelo se inicializa y se carga en la memoria (RAM o VRAM) una sola vez al arrancar el servicio. El usuario final solo experimenta la latencia de la generación de voz tras enviar su petición. Sumar el tiempo de carga a la síntesis falsearía la percepción de velocidad del sistema en un escenario real.
2.  **Por qué se descarta la primera síntesis:** La primera ejecución (calentamiento) incluye tareas estructurales únicas que el sistema no repetirá en las siguientes llamadas. Estas incluyen la compilación de grafos computacionales (típico en PyTorch u ONNX), la reserva inicial de *buffers* de memoria y la activación de cachés de hardware. Descartar esta medición evita penalizar la estadística con tiempos de inicialización diferida.
3.  **Por qué se usa la mediana:** El tiempo de procesamiento en un equipo personal está sujeto a fluctuaciones provocadas por el sistema operativo (procesos en segundo plano, notificaciones, gestión térmica). Estas fluctuaciones generan valores atípicos donde una frase concreta puede tardar el doble sin motivo aparente. La media aritmética es muy sensible a estos picos y se vería artificialmente inflada. La mediana (el valor central de la muestra ordenada) es una métrica mucho más robusta que aísla los extremos y refleja fielmente la experiencia habitual de velocidad.

## 4. Control prosódico

Resultados de las tareas 2.1 a 2.5, con sus figuras.

- Linealidad de la velocidad
- Determinismo y variabilidad
- Modalidad: ¿pregunta el sistema cuando debe?
- Foco: ¿se puede enfatizar una palabra?
- Normalización de texto

## 5. Calidad

Inteligibilidad, MOS y consistencia de voz, con intervalos de confianza.

Separad el WER de habla normal del de cifras y siglas, y clasificad cada error
de esas dos categorías como artefacto de convención o fallo real de síntesis.

Indicad qué convención usasteis en la columna `leida` y por qué.

## 6. Discusión

Dónde coinciden las tres medidas de calidad, dónde no, y por qué creéis que
ocurre. Las contradicciones entre medidas son lo más interesante del informe.

## 7. Decisión y limitaciones

Qué sistema usaríais en el asistente de vuestro dominio y por qué, ponderando
calidad, velocidad, recursos y licencia. Qué limitaciones tiene vuestro estudio.

## 8. Reflexión ética

Entre 500 y 800 palabras, con al menos dos fuentes citadas.
