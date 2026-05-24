# Tabla de Referencia de Modelos de IA (Vigentes y Deprecados - 2026)

Este documento es una guía de referencia rápida para evitar el uso de identificadores de modelos o APIs obsoletas. Mantener esta lista actualizada previene fallos silenciosos de "Model Not Found" durante la ejecución.

---

## 1. Google Gemini (Google GenAI SDK / Vertex AI)

### Modelos Activos e Identificadores de API

| Nombre Comercial | Identificador API | Contexto (Tokens) | Estado | Propósito y Caso de Uso |
| :--- | :--- | :--- | :--- | :--- |
| **Gemini 3.5 Flash** | `gemini-3.5-flash` | 1M - 2M | **Activo (Flagship)** | El más rápido y optimizado para flujos agenticos rápidos y RAG en tiempo real. |
| **Gemini 2.5 Pro** | `gemini-2.5-pro` | 2M | **Activo** | Razonamiento profundo, codificación compleja y análisis masivo de documentos. |
| **Gemini 2.5 Flash** | `gemini-2.5-flash` | 1M | **Activo** | Excelente relación costo/rendimiento para procesamiento general. |
| **Gemini 2.5 Flash-Lite** | `gemini-2.5-flash-lite` | 1M | **Activo** | Ultra-rápido, de bajo costo y optimizado para tareas repetitivas y chats básicos. |
| **Gemini 1.5 Pro** | `gemini-1.5-pro` | 2M | **Legacy / Activo** | Modelo estable de la generación anterior. Úsalo solo si se requiere compatibilidad específica. |
| **Gemini 1.5 Flash** | `gemini-1.5-flash` | 1M | **Legacy / Activo** | Modelo de velocidad estable de la generación anterior. |

### Modelos Deprecados / Fuera de Servicio (⚠️ NO USAR)
* `gemini-1.0-pro`
* `text-bison-001`
* `chat-bison-001`

### 💻 Ejemplo de Uso de la API de Google (Nueva SDK `google-genai`)
```python
# Instalar: pip install google-genai
from google import genai

client = genai.Client()
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Explica RAG en una frase.',
)
print(response.text)
```

---

## 2. OpenAI (GPT & Reasoning Models)

### Modelos Activos e Identificadores de API

| Nombre Comercial | Identificador API | Contexto (Tokens) | Estado | Propósito y Caso de Uso |
| :--- | :--- | :--- | :--- | :--- |
| **GPT-5.5** | `gpt-5.5` | 128k | **Activo (Flagship)** | Modelo general más avanzado con capacidades agentes nativas y baja alucinación. |
| **GPT-5.5 Instant** | `gpt-5.5-instant` | 128k | **Activo** | El modelo general rápido por excelencia, reemplazo directo de gpt-4o-mini. |
| **GPT-4o** | `gpt-4o` | 128k | **Legacy / Activo** | Excelente en visión, traducción y razonamiento multimodal general. |
| **GPT-4o-mini** | `gpt-4o-mini` | 128k | **Legacy / Activo** | El caballo de batalla clásico de bajo costo para tareas sencillas. |
| **o1** | `o1` | 128k | **Activo (Razonamiento)** | Razonamiento avanzado paso a paso. Ideal para matemáticas y programación compleja. |
| **o1-mini** | `o1-mini` | 128k | **Activo (Razonamiento)** | Razonamiento rápido y codificación a menor coste que o1. |
| **o3-mini** | `o3-mini` | 128k | **Activo (Razonamiento)** | Modelo de razonamiento rápido y eficiente. |

### Modelos Deprecados / Fuera de Servicio (⚠️ NO USAR)
* `gpt-3.5-turbo` (Deprecado/obsoleto, usar `gpt-4o-mini` o `gpt-5.5-instant`)
* `text-davinci-003`
* `gpt-4-turbo-preview`

### 💻 Ejemplo de Uso de la API de OpenAI
```python
# Instalar: pip install openai
import openai

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-5.5-instant",
    messages=[
        {"role": "user", "content": "Explica la diferencia entre L2 e Inner Product."}
    ]
)
print(response.choices[0].message.content)
```

---

## 3. Anthropic (Claude)

### Modelos Activos e Identificadores de API

| Nombre Comercial | Identificador API | Contexto (Tokens) | Estado | Propósito y Caso de Uso |
| :--- | :--- | :--- | :--- | :--- |
| **Claude Opus 4.7** | `claude-3-opus-4.7` | 1M | **Activo (Flagship)** | Máxima inteligencia en codificación, análisis de sistemas y tareas complejas de razonamiento. |
| **Claude Sonnet 4.6** | `claude-3-sonnet-4.6` | 200k | **Activo** | Excelente modelo diario para código, generación de contenido y agentes. |
| **Claude 3.5 Sonnet (v2)** | `claude-3-5-sonnet-latest` | 200k | **Legacy / Activo** | Modelo clásico equilibrado de alto rendimiento en lógica y código. |
| **Claude 3.5 Haiku** | `claude-3-5-haiku-latest` | 200k | **Legacy / Activo** | Modelo extremadamente rápido, ideal para flujos RAG masivos y categorización. |

### Modelos Deprecados / Fuera de Servicio (⚠️ NO USAR)
* `claude-instant-1.2`
* `claude-2.1`
* `claude-2.0`

### 💻 Ejemplo de Uso de la API de Anthropic
```python
# Instalar: pip install anthropic
import anthropic

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=1000,
    temperature=0,
    system="Eres un asistente técnico.",
    messages=[
        {"role": "user", "content": "Dame una función en Python para similitud de coseno."}
    ]
)
print(message.content[0].text)
```
