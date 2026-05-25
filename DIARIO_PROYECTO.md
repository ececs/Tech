# Diario del Proyecto & Sincronización entre Modelos

Este documento sirve como bitácora y canal de comunicación técnica entre los modelos de IA (**Gemini 3.5 Flash** y **Claude Sonnet 4.6**) que colaboran en la resolución de la prueba técnica de **TechStream** para el rol de Técnico de IA.

> [!IMPORTANT]
> **Instrucciones para el modelo entrante**: Lee este archivo completo para comprender el estado actual del desarrollo, las decisiones de diseño arquitectónico tomadas, y las tareas detalladas a implementar en la fase actual. Al finalizar tu trabajo, actualiza este archivo con tus progresos y decisiones tomadas.

---

## 📌 Estado General del Proyecto

* **Tiempo Máximo Estimado**: 12 Horas de Trabajo Activo.
* **Fase Actual**: Iniciando **Fase 2: Ingesta y Modelado en PyTorch**.
* **Modelo Entrante**: **Claude Sonnet 4.6 (Claude Code)**.

---

## 🛠️ Fase 1: Generación de Datos & EDA (Completada por Gemini 3.5 Flash)

### Qué se ha construido:
1. **[generate_data.py](file:///Users/daldo/VsCode/Tech/generate_data.py):** Script de simulación física de telemetría de servidores.
2. **[dataset_servidores.csv](file:///Users/daldo/VsCode/Tech/dataset_servidores.csv):** Dataset sintético con 10,000 registros secuenciales de series temporales.
3. **[eda_telemetria.ipynb](file:///Users/daldo/VsCode/Tech/eda_telemetria.ipynb):** Jupyter Notebook interactivo que analiza el comportamiento y las distribuciones de las variables de telemetría.
4. **[run_eda.py](file:///Users/daldo/VsCode/Tech/run_eda.py):** Script de verificación automática del EDA. Genera los gráficos PNG guardados en disco:
   * `distribution_clases.png`: Gráfico del desbalanceo de clases.
   * `matriz_correlacion.png`: Matriz de correlación de Pearson.
   * `distribucion_variables_fallo.png`: Distribución térmica e histogramas de sobrecarga de memoria por estado de fallo.

### Decisiones de Diseño y Hallazgos Clave:
* **Modelo Térmico de la CPU**: La temperatura de la CPU (`cpu_temp`) se modeló basándose en la Ley de Enfriamiento de Newton. Se calienta dinámicamente con el uso sostenido de CPU y se disipa hacia la temperatura ambiente del rack ($35^\circ\text{C}$). Esto crea una inercia térmica en el tiempo (el calor se acumula).
* **Desbalanceo de Clases**: De los 10,000 registros, **481 son fallos (4.81% del dataset)** y **9,519 son normales (95.19%)**.
  * *Acción obligatoria*: No se puede usar `Accuracy` como métrica de optimización. Se debe evaluar el modelo con **F1-Score**, **Precision** y **Recall**.
* **Lógica del Fallo**: Un fallo (`failure = 1`) ocurre con un 80% de probabilidad si la temperatura supera los $85^\circ\text{C}$ y el uso de memoria es $> 90\%$ durante **3 o más pasos consecutivos**. También hay un 5% de ruido estocástico de fallos aleatorios.
  * *Acción obligatoria*: Al existir dependencia secuencial de 3 pasos, la red neuronal necesita contexto temporal. Se usará una **ventana deslizante de tamaño 5** en la ingesta para que el modelo MLP reciba la evolución reciente de los sensores en lugar de una muestra aislada.

---

## ⚙️ Instrucciones para Claude Sonnet 4.6 (Fase 2: Ingesta y Modelado)

Claude, tu objetivo en esta fase es implementar el pipeline de datos, la arquitectura del modelo de PyTorch y el bucle de entrenamiento robusto. Sigue estrictamente estas directrices técnicas:

### 1. Requisitos Técnicos Obligatorios:
* **PEP 8 & Anotaciones de Tipo**: Todo el código nuevo debe cumplir con PEP 8 y estar 100% tipado (`type hints`).
* **Docstrings**: Cada módulo, clase y función debe contar con docstrings claros detallando parámetros, retornos y excepciones.
* **Logging Estructurado**: No uses `print()`. Utiliza el módulo `logging` de Python.
* **Dispositivo de Cómputo**: Implementar compatibilidad nativa con Apple Silicon (`torch.device("mps")`) con fallback automático a `cuda` o `cpu`.
* **Workaround de OpenMP**: Para evitar fallos silenciosos de segmentación en macOS con FAISS y PyTorch, añade las siguientes líneas al inicio de todos los archivos ejecutables:
  ```python
  import os
  os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
  os.environ["OMP_NUM_THREADS"] = "1"
  ```

---

### 2. Tareas Detalladas de la Fase 2:

#### 📋 Fase 2.A: Ingesta de Datos (`src/dataset.py`)
* Crear la clase `ServidorDataset` heredando de `torch.utils.data.Dataset`.
* Cargar el CSV [dataset_servidores.csv](file:///Users/daldo/VsCode/Tech/dataset_servidores.csv) y separar características (`cpu_usage`, `mem_usage`, `network_traffic`, `cpu_temp`) y etiqueta (`failure`).
* Escalar las características en el rango $[0, 1]$ usando `MinMaxScaler` o `StandardScaler` de `scikit-learn`. **Importante**: Guarda los parámetros del escalador en disco (o exporta el escalador con `pickle`/`joblib`) para poder usarlos en el guion de inferencia y pruebas futuras.
* Implementar la lógica de ventana deslizante:
  * Para una fila dada en el índice $t$, la entrada del modelo debe ser la concatenación de las características desde $t-\text{window\_size}+1$ hasta $t$ (donde $\text{window\_size} = 5$).
  * El tamaño del tensor de entrada por muestra debe ser `[window_size * num_features]` (es decir, $5 \times 4 = 20$ dimensiones aplanadas).
  * La etiqueta del target será el valor de `failure` en el instante $t$.
* Crear funciones para dividir el dataset en conjuntos de **Entrenamiento (70%)**, **Validación (15%)** y **Prueba (15%)** respetando la naturaleza temporal de las series (división secuencial o por bloques, evitando leaks de información temporales).
* Configurar los `DataLoader` de PyTorch. Utiliza `WeightedRandomSampler` o ajusta los lotes para gestionar el desbalanceo.

#### 🧠 Fase 2.B: Arquitectura del Modelo (`src/model.py`)
* Definir la clase `AnomalyDetectorMLP` heredando de `nn.Module`.
* Estructura recomendada:
  * Capa de entrada lineal: `20 -> 64` neuronas.
  * `BatchNorm1d` -> activación `ReLU` -> `Dropout(0.3)`.
  * Capa oculta lineal: `64 -> 32` neuronas.
  * `BatchNorm1d` -> activación `ReLU` -> `Dropout(0.3)`.
  * Capa de salida lineal: `32 -> 1` neurona.
* **Nota Crucial**: No apliques Sigmoide en la salida del modelo. Devuelve los logits directos de la última capa lineal. Esto permite usar la función de pérdida numéricamente estable `BCEWithLogitsLoss`.

#### 🏋️ Fase 2.C: Bucle de Entrenamiento (`src/train.py`)
* Implementar la función de pérdida `BCEWithLogitsLoss` configurando el parámetro `pos_weight` con el factor de balanceo calculado a partir de la proporción de clases (ej. $\text{peso} = \frac{\text{negativos}}{\text{positivos}} = \frac{9519}{481} \approx 19.79$) para compensar el desbalanceo del 4.81% de fallos.
* Optimizar con `Adam` usando un decaimiento de peso razonable (`weight_decay = 1e-4`).
* Programar un bucle de entrenamiento limpio:
  * Registro detallado de pérdida de entrenamiento y validación por época.
  * Registro de la métrica F1-score en validación al final de cada época.
  * Implementar **Early Stopping** (paciencia = 5 épocas monitoreando la pérdida de validación o el F1-score de validación) para evitar sobreentrenamiento.
  * Guardar el mejor estado del modelo en `/Users/daldo/VsCode/Tech/best_model.pth`.
  * Exportar y guardar la gráfica de las curvas de entrenamiento (Pérdida vs Época) en `training_curves.png`.

---

---

## ✅ Fase 2: Ingesta y Modelado (Completada por Claude Opus 4.7)

### Qué se ha construido:
1. **[src/dataset.py](file:///Users/daldo/VsCode/Tech/src/dataset.py)** — `ServidorDataset` con ventana deslizante causal de tamaño 5 (entrada plana de 20 dim), split temporal secuencial 70/15/15 sin shuffle, `MinMaxScaler` ajustado **solo en train** (evita data leakage temporal) y persistido en `artifacts/scaler.joblib` vía `joblib`. `DataLoader` de entrenamiento con `shuffle=True`; balanceo de clase delegado al `pos_weight` de la pérdida.
2. **[src/model.py](file:///Users/daldo/VsCode/Tech/src/model.py)** — `AnomalyDetectorMLP` 20→64→32→1 con `BatchNorm1d` + `ReLU` + `Dropout(0.3)` en cada bloque oculto. La salida son **logits puros** (sin sigmoide) para usarse con `BCEWithLogitsLoss`.
3. **[src/train.py](file:///Users/daldo/VsCode/Tech/src/train.py)** — Bucle de entrenamiento ejecutable con `python -m src.train`: detección automática de dispositivo MPS→CUDA→CPU, workaround OpenMP, semillas reproducibles, `BCEWithLogitsLoss` con `pos_weight` calculado dinámicamente y ajustable vía `--pos-weight-scale`, `Adam` con `--lr`/`--weight-decay`/`--dropout` configurables, **Early Stopping** monitorizando F1 de validación, logging estructurado por época, checkpoint del mejor estado en `best_model.pth` (incluye `decision_threshold` y `effective_pos_weight`), y exportación de la gráfica dual en `training_curves.png`.
4. **[src/sweep.py](file:///Users/daldo/VsCode/Tech/src/sweep.py)** — Barrido de hiperparámetros (`python -m src.sweep`) sobre el producto cartesiano de `lr × dropout × hidden_dims × pos_weight_scale × batch_size`. Para cada modelo entrenado, calcula el F1 con threshold óptimo sobre la curva precision-recall de validación, no con threshold fijo en 0.5. Persiste resultados en `sweep_results.jsonl` (append atómico para resistir crashes). Compatible MPS/CUDA/CPU.

### Decisiones de Diseño y Hallazgos Clave:
* **Decisión sobre el sampler ponderado**: la primera iteración combinaba `WeightedRandomSampler` con `pos_weight=20.2`. Empíricamente eso saturaba el modelo en falsos positivos (recall=1.0, precision≈0.05). Se eliminó el sampler y se mantiene únicamente el `pos_weight` en la pérdida (alineado con la indicación del diario).
* **Bug detectado en MPS**: `tensor.to(device, non_blocking=True)` provocaba corrupción de valores en MPS (la `loss.item()` devolvía valores absurdos del orden de 1e23/−1e23). Se eliminó `non_blocking=True` y el problema se resolvió. **Acción para futuros agentes**: no usar `non_blocking=True` en transferencias hacia `mps`.
* **Resultado tras el barrido de hiperparámetros (162 configs en MPS, 16 min)**: la configuración ganadora es `lr=3e-3, dropout=0.20, hidden=(64,32), pos_weight_scale=0.25, batch_size=64, threshold=0.71`. Con ella, **`val_f1 = 0.8545` (precision = 0.85, recall = 0.85)**. Hallazgos clave:
  * El `pos_weight = neg/pos = 20.2` del diario era demasiado agresivo: el modelo saturaba en falsos positivos. El sweep encuentra que `pos_weight ≈ 5` (scale 0.25) es óptimo.
  * El **umbral de decisión óptimo es ≈ 0.71**, no 0.5. Calcularlo sobre la curva PR de validación. `train.py` ahora persiste `decision_threshold` en el checkpoint.
  * La arquitectura `(64, 32)` del diario sí es óptima — más capacidad no aporta.
  * Resultados en `sweep_results.jsonl`; código en `src/sweep.py`.
* **Reproducibilidad**: `set_seed(42)` siembra `random`, `numpy` y `torch`; `MinMaxScaler` ajustado solo en train evita leakage.
* **Artefactos generados**:
  * `artifacts/scaler.joblib` (escalador para inferencia).
  * `best_model.pth` (estado, dimensiones y `best_val_f1` para reconstrucción).
  * `training_curves.png` (loss train/val + métricas de val).

---

---

## ✅ Fase 3: Evaluación y Robustez (Completada por Claude Opus 4.7)

### Qué se ha construido:
1. **[src/evaluate.py](file:///Users/daldo/VsCode/Tech/src/evaluate.py)** — `python -m src.evaluate` carga `best_model.pth` (reconstruyendo la arquitectura desde la metadata del checkpoint), reusa `build_dataloaders()` para obtener el test split idéntico al de entrenamiento, ejecuta inferencia con el umbral calibrado de Fase 2, imprime métricas + `classification_report` de sklearn y guarda la matriz de confusión anotada (TN/FP/FN/TP) en `test_confusion_matrix.png`. Acepta `--threshold` y `--checkpoint` para experimentar.
2. **[tests/test_model.py](file:///Users/daldo/VsCode/Tech/tests/test_model.py)** — 9 tests con `pytest`:
   * `test_model_io_shapes` (parametrizado en batch_size ∈ {1, 8, 32, 128}) verifica forma `[B, 20] → [B, 1]`, dtype `float32` y logits finitos.
   * `test_model_rejects_wrong_input_dim` verifica que `Linear` falla con input 19 dim.
   * `test_dataset_scaling_and_window` verifica que `ServidorDataset` produce ventanas en `[0, 1]`, con shape `[20]`, y que la ventana es **causal** (la primera muestra coincide con las primeras 5 filas escaladas, no con un futuro).
   * `test_dataset_rejects_short_split`, `test_temporal_split_is_sequential`, `test_fit_scaler_uses_only_training_rows`.
   * Resultado: **9/9 passed en 2.5 s**.

### Resultados Finales sobre el Test Split (15 % temporal, 1496 ventanas, 100 positivos):

| Métrica  | Validación | **Test** |
|----------|-----------|----------|
| F1-score | 0.8545 | **0.8384** |
| Precision| 0.8545 | **0.8469** |
| Recall   | 0.8545 | **0.8300** |
| Accuracy | —     | **0.9786** |

| Matriz de confusión | Pred = 0 | Pred = 1 |
|---|---|---|
| **True = 0** | TN = 1381 | FP = 15 |
| **True = 1** | FN = 17 | TP = 83 |

* **Generalización**: el gap val→test es de solo **−0.016 en F1**, lo cual descarta sobreajuste. Falsa alarma (FPR) en producción = 1.07 %; pérdida de detección = 17 %.
* **Artefacto**: `test_confusion_matrix.png`.

---

## 🧪 Fase 4 (Siguientes Pasos de Referencia)

* **Fase 3 (Pruebas y Evaluación)**:
  * Claude deberá crear `src/evaluate.py` para cargar `best_model.pth`, realizar inferencia en el test split, y generar métricas finales (F1-score, Precision, Recall y matriz de confusión).
  * Claude escribirá pruebas unitarias en `tests/test_model.py` usando `pytest` para verificar dimensiones de tensores de la red neuronal.
* **Fase 4 (Documentación & Guion)**:
  * Gemini 3.5 Flash redactará el `README.md` final del repositorio y el guion del vídeo técnico explicativo (`GUION_VIDEO.md`) de 10 minutos para Airbus Helicopters y TechStream.
