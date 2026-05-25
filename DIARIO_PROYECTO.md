# Diario Técnico del Proyecto

Este documento resume las decisiones técnicas más importantes tomadas durante el desarrollo del reto, pero en una versión apta para publicación en GitHub.

---

## 1. Generación de datos

Se construyó un generador sintético de telemetría de servidores con:

* uso de CPU y memoria con dinámica autorregresiva,
* tráfico de red correlacionado con CPU,
* temperatura de CPU modelada con una versión discreta de la ley de enfriamiento de Newton,
* lógica causal de fallo basada en sobrecalentamiento sostenido y presión de memoria.

Objetivo: que el dataset no fuera ruido aleatorio, sino una serie temporal con dependencias físicas plausibles.

---

## 2. Pipeline de datos

Decisiones clave:

* split temporal secuencial 70 / 15 / 15 para evitar leakage,
* `MinMaxScaler` ajustado solo en entrenamiento,
* persistencia del scaler para reuso en evaluación e inferencia,
* ventana deslizante causal de tamaño 5 para capturar contexto temporal reciente.

Resultado: el modelo recibe 20 variables por muestra (`5 pasos × 4 features`).

---

## 3. Modelo base entregable

El primer camino de entrega se cerró con un **MLP** porque era la opción más robusta y directa para cumplir el reto con garantías.

Arquitectura (3,649 parámetros):

* entrada de 20 dimensiones,
* capas ocultas 64 y 32,
* `BatchNorm1d`,
* `ReLU`,
* `Dropout`,
* salida logística con `BCEWithLogitsLoss`.

Además:

* early stopping,
* threshold calibrado en validación,
* checkpoint del mejor estado,
* suite de tests automatizados.

---

## 4. Hallazgos del entrenamiento

Hallazgos relevantes:

* el problema está desbalanceado, por lo que la métrica prioritaria es F1, no accuracy;
* el `pos_weight` crudo (o bruto) era de 18.82 y se suavizó durante el sweep a un pos_weight efectivo de 4.70 (escala 0.25);
* el threshold óptimo no fue 0.5, sino aproximadamente 0.71;
* para este modelo pequeño, la diferencia entre aceleradores vino más por coste de transferencia que por capacidad de cómputo.

---

## 5. Comparativa MLP vs GRU

Tras cerrar la entrega con el MLP, se añadió una variante **GRU** (3,681 parámetros) como extensión.

Conclusión principal:

* con una secuencia tan corta (`seq_len = 5`), el sesgo recurrente no compensa claramente;
* la GRU mejora ligeramente el recall, pero empeora el equilibrio global por el aumento de falsos positivos;
* el **MLP sigue siendo la mejor entrega base** para este escenario.

Esto se implementó manteniendo ambos entrypoints separados:

* `src.train` para MLP
* `src.train_gru` para GRU

y compartiendo solo la lógica interna común de entrenamiento.

---

## 6. Evaluación e inferencia

Se reforzó la coherencia offline/online:

* evaluación reutiliza el scaler persistido en lugar de recalcularlo,
* el checkpoint guarda la configuración real del modelo,
* la app web usa exactamente los mismos artefactos del pipeline.

---

## 7. RAG y demo web

Se construyeron dos capas adicionales:

### RAG CLI

* índice FAISS local persistido,
* embeddings con `sentence-transformers`,
* generación sobre Ollama,
* citas de documentación con `source` y `start_line`,
* fallback a modelo local si el servidor remoto no responde.

### FastAPI demo

Incluye:

* snapshot inference con sliders,
* batch CSV upload,
* gráfico temporal,
* descarga de resultados,
* chat RAG integrado.

---

## 8. Estado actual

El proyecto ya no es solo un entrenamiento aislado, sino una entrega bastante completa:

* pipeline reproducible,
* evaluación robusta,
* dos variantes de modelo,
* demo web,
* RAG documental,
* documentación y utilidades de presentación.

---

## 9. Limitaciones reconocidas

* El modo slider de la web es aproximado porque repite la misma lectura 5 veces.
* El batch mode es más representativo que el snapshot mode.
* El chat RAG depende de disponibilidad de Ollama o de un fallback local.
* La persistencia de sesiones es local porque el objetivo del reto era una demo técnica, no un sistema multiusuario en producción.
