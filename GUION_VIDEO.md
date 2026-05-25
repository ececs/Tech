# Guion para el Vídeo Explicativo: Detección de Anomalías (10 Minutos)

Este documento contiene la estructura detallada paso a paso y el guion narrativo para que grabes tu vídeo de presentación técnica para **TechStream** y el equipo de selección.

---

## ⏱️ Estructura Temporal del Vídeo

| Sección | Tiempo | Foco Visual | Mensaje Clave |
| :--- | :---: | :--- | :--- |
| **1. Introducción & CV** | 0:00 - 1:30 | Tu cámara web / Diapositiva inicial | Presentación, encaje con el perfil y contextualización. |
| **2. Generación Física & EDA** | 1:30 - 3:30 | VS Code (`generate_data.py`) y gráficos PNG | Explicación del modelo térmico y el desbalanceo del 5%. |
| **3. Pipeline & Ventana Causal** | 3:30 - 5:00 | VS Code (`src/dataset.py`) | Split temporal y lógica de la ventana de 5 pasos. |
| **4. Arquitectura MLP PyTorch** | 5:00 - 6:30 | VS Code (`src/model.py` y `src/train.py`) | Modelado, logits, BCE y regularización. |
| **5. Sweep de Hiperparámetros** | 6:30 - 8:30 | Gráfico `training_curves.png` y consola | Resultados del sweep, calibración del umbral y cuello de botella I/O. |
| **6. Evaluación & Conclusión** | 8:30 - 10:00 | Matriz de confusión y terminal de tests | Generalización (test F1=0.83), robustez y cierre. |

---

## 🎙️ Guion Narrativo Paso a Paso

### 🎬 Sección 1: Introducción y Encaje del Perfil (0:00 - 1:30)
* **Acción Visual:** Muestra tu rostro en cámara o una diapositiva con tu nombre, título del puesto y tus datos profesionales.
* **Qué decir:**
  > *"Hola a todos. Mi nombre es Eudaldo Cal y en este vídeo voy a defender la resolución del reto técnico para la posición de Técnico de IA en TechStream.*
  >
  > *Antes de entrar en el código, me gustaría destacar que cuento con una sólida experiencia de 17 años operando infraestructuras industriales críticas en Veolia, lo cual me ha aportado una disciplina operativa total y tolerancia cero a errores. Además, he fundado mi propia consultora de automatización y orquestación de IA (D4Lab y TesIA), y actualmente estoy terminando el Grado en Ingeniería Informática. Este reto técnico aúna a la perfección mis dos grandes pasiones: la analítica de datos en infraestructuras críticas y el Deep Learning en PyTorch.*
  >
  > *Para resolver el problema, he diseñado un pipeline modular completo en PyTorch que va desde la simulación de telemetría basada en leyes físicas hasta la optimización de hiperparámetros y validación robusta del sistema. A continuación, les muestro el código."*

---

### 📊 Sección 2: Simulación de Sensores y EDA (1:30 - 3:30)
* **Acción Visual:** Abre en VS Code [generate_data.py](file:///Users/daldo/VsCode/Tech/generate_data.py) y muestra los gráficos `distribucion_variables_fallo.png` y `matriz_correlacion.png`.
* **Qué decir:**
  > *"Un aspecto crucial al trabajar con datos sintéticos es que tengan coherencia física. Si generamos ruido puramente aleatorio, el modelo no aprenderá relaciones reales de causa-efecto.*
  >
  > *Por ello, he implementado un generador basado en la **Ley de Enfriamiento de Newton** en `generate_data.py`. El uso de CPU y memoria siguen procesos autorregresivos realistas. El tráfico de red se correlaciona con la CPU, y la temperatura de la CPU acumula calor de forma exponencial según la carga de trabajo y disipa calor hacia el rack a $35^\circ\text{C}$.*
  >
  > *Un fallo se produce de forma lógica si hay sobrecarga de temperatura y memoria sostenida durante **3 o más minutos**. El ruido aleatorio de fallo se limitó al 0.1% para que el modelo realmente tuviera señal que aprender.*
  >
  > *El dataset resultante contiene **508 anomalías (5.08%)**, reflejando la realidad de un sistema desbalanceado. En la matriz de correlación del EDA podemos comprobar que la temperatura y la CPU están altamente correlacionadas con el fallo, lo que valida la coherencia física de los datos simulados."*

---

### 📋 Sección 3: Pipeline de Datos y Ventana Causal (3:30 - 5:00)
* **Acción Visual:** Muestra en pantalla el archivo [src/dataset.py](file:///Users/daldo/VsCode/Tech/src/dataset.py).
* **Qué decir:**
  > *"Para procesar los datos de forma robusta, he estructurado el código en `src/dataset.py`. Aquí he tomado tres decisiones críticas de diseño:*
  >
  > *Primero, realizamos un **split temporal secuencial (70% Train, 15% Val, 15% Test)** sin mezclar aleatoriamente el dataset. Esto es fundamental en series temporales para evitar el 'look-ahead bias' o fuga de información del futuro hacia el pasado.*
  >
  > *Segundo, ajustamos el `MinMaxScaler` **únicamente sobre el conjunto de entrenamiento** y lo guardamos en disco como un artefacto. Así garantizamos que las transformaciones aplicadas a validación y prueba se hagan con parámetros totalmente desconocidos para ellos.*
  >
  > *Tercero, dado que la temperatura tarda tiempo en acumularse antes de un fallo, una muestra aislada no tiene suficiente información. He implementado una **ventana deslizante causal de tamaño 5**. Esto significa que para predecir si el servidor falla en el minuto $t$, el modelo recibe el historial de los minutos $t-4$ a $t$, aplanando las variables en un vector de entrada de 20 dimensiones."*

---

### 🧠 Sección 4: Arquitectura MLP de PyTorch (5:00 - 6:30)
* **Acción Visual:** Abre [src/model.py](file:///Users/daldo/VsCode/Tech/src/model.py) y [src/train.py](file:///Users/daldo/VsCode/Tech/src/train.py).
* **Qué decir:**
  > *"Para la red neuronal, en `src/model.py` he diseñado un Multi-Layer Perceptron (MLP) denso en PyTorch. La entrada de 20 variables pasa por dos capas ocultas de 64 y 32 neuronas, aplicando normalización por lotes (`BatchNorm1d`), funciones de activación `ReLU` y regularización `Dropout(0.3)` para evitar el sobreentrenamiento.*
  >
  > *En la salida devolvemos los logits puros de la última capa lineal en lugar de aplicar una función sigmoide. Esto nos permite usar `BCEWithLogitsLoss` en `src/train.py`, que es mucho más estable numéricamente.*
  >
  > *Para entrenar el modelo, compensamos el desbalanceo del 5% de anomalías asignando un peso a la pérdida positiva (`pos_weight`) y configuramos un Early Stopping que detiene el entrenamiento de forma automática si el F1-Score en validación deja de mejorar durante 5 épocas, lo que previene que la red se sobreajuste."*

---

### 🏋️ Sección 5: Sweep de Hiperparámetros & Cuello de Botella I/O (6:30 - 8:30)
* **Acción Visual:** Muestra [src/sweep.py](file:///Users/daldo/VsCode/Tech/src/sweep.py) y abre el gráfico `training_curves.png`.
* **Qué decir:**
  > *"Diseñamos un script de barrido exhaustivo en `src/sweep.py` que evaluó **162 combinaciones** de hiperparámetros. Durante el sweep, calibramos dinámicamente el umbral de decisión sobre la curva Precision-Recall para maximizar el F1-Score en validación, en lugar de usar un umbral de 0.5.*
  >
  > *Aquí descubrimos un **hallazgo técnico muy valioso sobre el hardware**: Ejecutamos el sweep completo tanto en mi Mac con chip M4 (usando aceleración MPS) como en mi PC con una GPU dedicada NVIDIA RTX 4070 Super (usando CUDA). El sweep tardó **16.1 minutos en el Mac** y **17.8 minutos en el PC**.*
  >
  > *¿Por qué la 4070 Super fue marginalmente más lenta? Al ser un modelo muy pequeño de solo 3,000 parámetros, el tiempo necesario para transferir los tensores entre la CPU y la GPU a través del bus PCI-e supera con creces el tiempo de cálculo de la GPU. El entrenamiento era **I/O-bound** (cuello de botella de entrada/salida). En cambio, la memoria unificada del chip M4 de Apple Silicon manejó la transferencia de forma mucho más eficiente.*
  >
  > *El sweep óptimo seleccionó una arquitectura `(128, 64)`, redujo el peso positivo efectivo a **5.05** y determinó que el umbral de decisión perfecto era **0.71**, logrando catapultar el F1-Score de validación desde **0.64 hasta un excelente 0.854**."*

---

### 🧪 Sección 6: Evaluación de Test y Conclusión (8:30 - 10:00)
* **Acción Visual:** Abre `test_confusion_matrix.png` y ejecuta `pytest tests/` en la terminal para mostrar las pruebas unitarias en verde.
* **Qué decir:**
  > *"Finalmente, evaluamos el mejor modelo guardado en `best_model.pth` sobre el conjunto de prueba (los últimos 1,500 registros del servidor que el modelo jamás vio durante el entrenamiento). El resultado en el test split fue sobresaliente:*
  >
  > *Obtenemos un **F1-Score en test de 0.838**, con una Precision de **0.847** y un Recall de **0.830**. La bajada mínima respecto a validación descarta por completo el sobreentrenamiento.*
  >
  > *La matriz de confusión en pantalla refleja este gran desempeño: clasificamos correctamente casi todos los fallos térmicos y mantenemos los falsos positivos en niveles insignificantes, lo cual es de vital importancia en producción para no saturar a los operadores de mantenimiento.*
  >
  > *Adicionalmente, he creado una suite de 9 pruebas unitarias automáticas con pytest que garantizan la integridad de las dimensiones del pipeline y el modelo ante futuros despliegues.*
  >
  > *Como siguientes pasos para llevar esta solución a producción, sugiero explorar arquitecturas recurrentes de Deep Learning como LSTM o GRU, o capas convolucionales 1D. Esto nos permitiría explotar directamente la naturaleza secuencial temporal de los datos sin tener que aplanar la ventana deslizante.*
  >
  > *Muchas gracias por su atención y quedo a su disposición para cualquier pregunta."*

---

## 💡 Consejos para la Grabación:
* **Tono:** Mantén un tono dinámico, seguro y con rigor técnico.
* **Lente:** Mira a la cámara en la introducción y conclusión. Cuando compartas pantalla, haz zoom en el código de VS Code para que el texto sea legible.
* **Preparación:** Ejecuta `pytest tests/` una vez antes de grabar para que el comando se cargue rápido en el historial de la terminal.
