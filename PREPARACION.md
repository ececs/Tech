# Guía de Preparación: Reto Técnico de IA/ML (24 Horas)

Este documento sirve como hoja de ruta y preparación para el reto técnico de 24 horas. Considerando los requisitos del puesto, el reto probablemente se centrará en una de las siguientes tres áreas (o una combinación de ellas):

1. **Sistema RAG (Retrieval-Augmented Generation)**: Ingesta, chunking, generación de embeddings, almacenamiento en base de datos vectorial (local como FAISS/Chroma) y consulta con un LLM.
2. **Entrenamiento/Fine-tuning de Modelos**: Un pipeline en PyTorch para clasificar texto/imágenes o un script de fine-tuning ligero usando LoRA/QLoRA con Hugging Face.
3. **Diseño de Arquitectura de Redes Neuronales**: Implementación de un modelo personalizado (CNN, RNN o Transformer simple) y su ciclo de entrenamiento completo.

---

## 🛠️ Estructura Recomendada del Proyecto

Para resolver el reto de manera limpia, profesional y reproducible en producción:

```text
tech-challenge/
├── README.md             # Explicación del enfoque, cómo ejecutarlo y decisiones de diseño
├── requirements.txt      # Dependencias exactas
├── data/                 # Datos del reto (ignorar en git si son grandes)
├── notebooks/            # Exploración inicial (EDA) rápida
├── src/                  # Código fuente modular
│   ├── __init__.py
│   ├── data_pipeline.py  # Ingesta, limpieza y chunking
│   ├── model.py          # Definición de redes PyTorch o cargadores de LLMs
│   ├── vector_store.py   # Conexión y operaciones con DBs vectoriales
│   └── main.py           # Punto de entrada principal
└── tests/                # Pruebas unitarias básicas para demostrar robustez
```

---

## 🚀 Checklist para las 24 Horas

- [ ] **Leer todo el enunciado primero**: No programar de inmediato. Identificar las métricas de éxito (precisión, latencia, modularidad).
- [ ] **Entorno virtual limpio**: Crear un entorno virtual (`python -m venv venv`) y documentar cada librería en `requirements.txt`.
- [ ] **EDA rápido (Exploratory Data Analysis)**: Si te dan un dataset, haz un Jupyter Notebook rápido para analizar nulos, distribuciones y formatos.
- [ ] **MVP en 4 horas**: Construye el pipeline de inicio a fin lo antes posible, aunque sea con un modelo básico.
- [ ] **Métricas y Evaluación**: Define cómo medirás el éxito (ej. F1-score para clasificación, ROUGE/BLEU o Ragas para RAG).
- [ ] **Refactorización y Limpieza**: Pasa el código del notebook a módulos `.py` limpios, siguiendo buenas prácticas (PEP 8, type hints, docstrings).
- [ ] **Documentación impecable**: Un reto de 24h destaca por su `README.md`. Explica *por qué* elegiste cada solución y cuáles serían los siguientes pasos si tuvieras más tiempo.

---

## 📚 Plantillas Disponibles en `/templates`

He preparado tres plantillas base con el código estructurado para que puedas copiar, adaptar y acelerar tu desarrollo:

1. **`templates/rag_pipeline_template.py`**: Implementación completa de un flujo RAG (LangChain + FAISS + HuggingFace/OpenAI).
2. **`templates/pytorch_training_template.py`**: Pipeline completo de entrenamiento en PyTorch (Dataset personalizado, Red Neuronal y bucle de entrenamiento/validación con gráficas de rendimiento).
3. **`templates/llm_finetuning_template.py`**: Estructura para Fine-Tuning eficiente usando Hugging Face `transformers`, `peft` (LoRA) y `trl`.
