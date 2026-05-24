# Guía de Referencia Rápida: LLMs-from-scratch (Sebastian Raschka)

Para el reto técnico de mañana, es sumamente útil contar con referencias y estructuras de código base en PyTorch puro. He clonado el repositorio de forma local y privada en tu directorio de trabajo en la carpeta `LLMs-from-scratch/`. 

Esta carpeta está en tu `.gitignore` para no interferir con tu control de versiones público en GitHub, pero puedes abrir y buscar sus archivos desde tu editor de código.

A continuación, tienes un índice de acceso rápido a los componentes clave que te pueden pedir diseñar desde cero o ajustar:

---

## 📂 1. Preparación de Datos y Tokenización

Si la prueba requiere preprocesar texto para entrenar un modelo desde cero:
* **Tokenizador TikToken & Vocabulario**:
  * Archivo principal: `LLMs-from-scratch/ch02/01_main-chapter-code/ch02.ipynb`
  * Muestra cómo mapear texto a IDs de tokens, añadir tokens especiales (`<|endoftext|>`), y configurar el embedding de tokens y posición.
* **DataLoader de Ventana Deslizante (Sliding Window)**:
  * Archivo principal: `LLMs-from-scratch/ch02/01_main-chapter-code/dataloader.ipynb`
  * Contiene la clase `GPTDatasetV1` y el generador `create_dataloader_v1` en PyTorch. Es ideal para crear lotes de entrenamiento de texto donde la entrada es `x` y el objetivo es `y` (el siguiente token desplazado en 1).

---

## 🧠 2. Mecanismos de Atención y Arquitectura Transformer

Si te piden diseñar arquitecturas de redes neuronales específicas o capas de Transformers desde cero:
* **Causal Self-Attention (Atención con Máscara)**:
  * Archivo principal: `LLMs-from-scratch/ch03/01_main-chapter-code/ch03.ipynb`
  * Código para implementar la máscara causal que impide al modelo "mirar al futuro" durante el entrenamiento.
* **Multi-Head Attention (MHA) en PyTorch**:
  * Archivo principal: `LLMs-from-scratch/ch03/01_main-chapter-code/multihead-attention.ipynb`
  * Contiene la implementación de la clase `MultiHeadAttention` usando capas lineales eficientes.
* **Arquitectura GPT Completa**:
  * Archivo principal: `LLMs-from-scratch/ch04/01_main-chapter-code/gpt.py`
  * Clase `DummyGPTModel` y la implementación real del bloque de Transformer (`TransformerBlock`), capa de normalización (`LayerNorm`), y función de activación GELU.
* **Variantes de Atención Modernas (Bonus)**:
  * **Mixture of Experts (MoE)**: `LLMs-from-scratch/ch04/07_moe/` (múltiples redes feed-forward con compuertas de enrutamiento).
  * **Multi-Head Latent Attention (MLA - estilo DeepSeek)**: `LLMs-from-scratch/ch04/05_mla/`.

---

## 📈 3. Entrenamiento y Ajuste (Fine-Tuning)

Si la prueba implica ajustar modelos existentes o entrenar modelos locales:
* **Bucle de Entrenamiento y Pérdida de Causal LM**:
  * Archivo principal: `LLMs-from-scratch/ch05/01_main-chapter-code/gpt_train.py`
  * Implementación estándar de la pérdida de entropía cruzada (`CrossEntropyLoss`) y el bucle para entrenar y evaluar en conjuntos de entrenamiento/validación.
* **Ajuste para Clasificación de Texto (Sentiment Analysis)**:
  * Archivo principal: `LLMs-from-scratch/ch06/01_main-chapter-code/`
  * **Estrategia**: Carga un modelo GPT preentrenado, reemplaza el cabezal de salida (`out_head`) por una capa lineal adaptada al número de clases, congela las capas iniciales si es necesario, y realiza el entrenamiento sobre el dataset etiquetado.
* **Ajuste para Seguimiento de Instrucciones (Instruction Fine-Tuning / SFT)**:
  * Archivo principal: `LLMs-from-scratch/ch07/01_main-chapter-code/`
  * Muestra cómo formatear datasets en formato de instrucción (Alpaca/ShareGPT) y entrenar al modelo para responder preguntas directas.

---

## 🚀 4. Técnicas Avanzadas y Eficiencia (PEFT / LoRA)

* **LoRA (Low-Rank Adaptation) desde cero en PyTorch**:
  * Directorio: `LLMs-from-scratch/appendix-D/01_main-chapter-code/`
  * Muestra cómo implementar capas lineales con adaptadores LoRA (`LinearWithLoRA`) en PyTorch puro sin depender de la librería `peft` de Hugging Face. Es un recurso espectacular si te piden demostrar conocimiento profundo de cómo funciona LoRA a nivel matemático.
* **Ajuste de Preferencias (DPO - Direct Preference Optimization)**:
  * Directorio: `LLMs-from-scratch/ch07/04_preference-tuning-with-dpo/`
  * Implementación de optimización de preferencias para alinear respuestas del modelo.
