# Catálogo de Proyectos y Componentes Reutilizables

Este catálogo resume los proyectos desarrollados anteriormente en tu entorno de trabajo, detallando sus funcionalidades, pila tecnológica y componentes específicos que podemos reutilizar o adaptar rápidamente durante la resolución del reto técnico de mañana.

---

## 1. Proyecto: D4-Ticket-AI (Proyecto Final DAW)
* **Directorio**: `/Users/daldo/VsCode/D4-ticket-AI` (y `/Users/daldo/VsCode/Daw Proyecto Final`)
* **Propósito**: Plataforma colaborativa de gestión de incidencias con un agente conversacional de IA integrado en tiempo real y base de conocimiento RAG.
* **Pila Tecnológica**: Python, FastAPI, SQLAlchemy, PostgreSQL (pgvector), `trafilatura` (scraping), `boto3` (MinIO/S3), `asyncpg` (LISTEN/NOTIFY), LangGraph, `psycopg-pool`, Next.js (frontend).

### 🛠️ Componentes y Patrones de Código Reutilizables:
* **[embedding_service.py](file:///Users/daldo/VsCode/D4-ticket-AI/backend/app/services/embedding_service.py)**:
  * Generación asíncrona de embeddings con la API de Google (`gemini-embedding-2`) usando `httpx.AsyncClient`.
  * Parámetro `outputDimensionality` truncado a 768 mediante Matryoshka Representation Learning (ideal para ahorrar RAM y optimizar búsquedas).
* **[knowledge_service.py](file:///Users/daldo/VsCode/D4-ticket-AI/backend/app/services/knowledge_service.py)**:
  * Función de troceado (`_chunk_text`) basada en límites de párrafos con control de solapamiento (*overlap*).
  * Web scraping asíncrono no bloqueante con `trafilatura` mediante `asyncio.to_thread`.
  * Ingestión idempotente (borra e inserta).
  * Búsqueda de similitud de coseno en base de datos con SQLAlchemy (`cosine_distance`) y **fallback automático a coincidencia textual ILIKE** si no se dispone de embeddings.
* **[checkpoint.py](file:///Users/daldo/VsCode/D4-ticket-AI/backend/app/ai/checkpoint.py)**:
  * Checkpointer de base de datos persistente para LangGraph con `AsyncPostgresSaver` y `psycopg_pool.AsyncConnectionPool`.
  * Fallback gracioso a modo *stateless* si la base de datos no está disponible.
* **[main.py](file:///Users/daldo/VsCode/D4-ticket-AI/backend/app/main.py)**:
  * Estructura de ciclo de vida (`lifespan`) moderna en FastAPI para inicializar bases de datos, checkpointers y sockets.
  * Bucle de escucha `LISTEN/NOTIFY` asíncrono con `asyncpg` para comunicación WebSocket en tiempo real.

---

## 2. Proyecto: Copilot Financiero
* **Directorio**: `/Users/daldo/VsCode/Copilot financiero`
* **Propósito**: Asistente de IA para el análisis de portafolios, rebalanceo financiero automatizado y evaluación de riesgos de inversión.
* **Pila Tecnológica**: Python, FastAPI, React Native (Expo, Nativewind), LangGraph, Pydantic v2.

### 🛠️ Componentes y Patrones de Código Reutilizables:
* **[state.py](file:///Users/daldo/VsCode/Copilot%20financiero/backend/src/copilot/graphs/state.py)**:
  * Patrón de diseño de estado mutable para LangGraph utilizando un modelo de **Pydantic v2** (`GraphState(BaseModel)`) en lugar de `TypedDict`. Esto añade validación de tipos e inicialización limpia de objetos de negocio de manera automática.
* **[advice_graph.py](file:///Users/daldo/VsCode/Copilot%20financiero/backend/src/copilot/graphs/advice_graph.py)**:
  * Flujo de trabajo de grafo complejo y determinista: `Precalcular` -> `Generar` -> `Validar` -> `Fin / Reintento / Fallback`.
  * Lógica de reintentos controlada (`attempts` frente a `max_attempts`) y rutas condicionales basadas en la respuesta del evaluador.

---

## 3. Proyecto: Tutor de Inglés Primaria (Sparky)
* **Directorio**: `/Users/daldo/VsCode/tutor_ingles_primaria`
* **Propósito**: Chatbot interactivo y tutor socrático de inglés para niños de primaria con capacidades de visión y análisis de tareas.
* **Pila Tecnológica**: Python, LangGraph, LangChain, ChatVertexAI (Gemini 2.5 Flash Lite).

### 🛠️ Componentes y Patrones de Código Reutilizables:
* **[sparky_graph.py](file:///Users/daldo/VsCode/tutor_ingles_primaria/sparky_graph.py)**:
  * Orquestación dinámica y enrutado condicional basado en la entrada del usuario (si contiene una imagen o texto activa el flujo visual, si es texto normal activa el flujo socrático).
  * Prompts de tutoría socrática para guiar el aprendizaje sin entregar soluciones directas.
  * Nodo "Tone Shaper" para reescribir respuestas complejas en un tono amigable, motivador y cargado de emojis, ideal para interfaces de usuario interactivas.

---

## 4. Proyecto: d4lab-web-clean
* **Directorio**: `/Users/daldo/VsCode/d4lab-web-clean`
* **Propósito**: Sitio web corporativo de tu estudio de IA y desarrollo D4Lab.
* **Pila Tecnológica**: HTML, Tailwind CSS, Vanilla JS.

### 🛠️ Componentes y Patrones de Código Reutilizables:
* **[chat-widget.js](file:///Users/daldo/VsCode/d4lab-web-clean/chat-widget.js)**:
  * Widget de chat dinámico en Vanilla JS integrable en cualquier página web mediante una sola etiqueta `<script>`. Excelente si el reto requiere acoplar una interfaz gráfica al modelo de IA en el frontend de forma rápida.
