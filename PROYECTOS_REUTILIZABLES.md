# Catálogo de Patrones Reutilizables

Este documento resume patrones y componentes reutilizables tomados de proyectos previos, pero sin exponer rutas locales ni referencias privadas.

---

## 1. Plataforma de tickets con RAG

**Propósito**: gestión de incidencias con base de conocimiento y asistente conversacional.

**Patrones útiles**:
* Servicio de embeddings asíncrono con `httpx.AsyncClient`.
* Chunking por párrafos con overlap controlado.
* Scraping no bloqueante con `asyncio.to_thread`.
* Ingesta idempotente y fallback de búsqueda textual si no hay embeddings.
* `lifespan` de FastAPI para inicialización ordenada de servicios compartidos.

---

## 2. Copilot financiero

**Propósito**: análisis de portafolios y recomendaciones automáticas.

**Patrones útiles**:
* Estado de grafo modelado con **Pydantic v2** en vez de `TypedDict`.
* Flujos deterministas con validación, reintentos y rutas de fallback.
* Separación clara entre capa de orquestación y capa de servicio.

---

## 3. Tutor conversacional

**Propósito**: chatbot educativo con rutas distintas para texto e imagen.

**Patrones útiles**:
* Enrutado dinámico según el tipo de entrada del usuario.
* Prompts socráticos que guían sin entregar la solución directamente.
* Reescritura final de tono para adaptar respuestas técnicas a UX amigable.

---

## 4. Widget web ligero

**Propósito**: integrar un chat en páginas estáticas sin frameworks.

**Patrones útiles**:
* Componente de chat en Vanilla JS integrable con una sola etiqueta `<script>`.
* UI mínima con estado local, fetch al backend y render incremental de mensajes.

---

## Conclusión

El valor de este catálogo no está en archivos concretos de un entorno local, sino en los patrones de diseño transferibles: `lifespan` en FastAPI, servicios compartidos, fallback controlado, orquestación por estado y frontends ligeros para demos técnicas.
