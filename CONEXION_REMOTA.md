# Configuración de Conexión Remota (MacBook Air M4 -> PC RTX 4070 Super)

Este documento resume los parámetros de red y los servicios configurados en tu PC para permitir el acceso y procesamiento remoto desde tu MacBook Air.

---

## 1. Conexión SSH (Consola y Desarrollo Remoto)

El servidor SSH de Windows está **activo** y configurado para aceptar conexiones en el puerto estándar (22).

* **IP del Servidor (PC)**: `192.168.31.181`
* **Usuario de Windows**: `eudal`
* **Puerto**: `22` (Confirmado: Abierto y respondiendo)
* **Comando de conexión (Terminal del Mac)**:
  ```bash
  ssh eudal@192.168.31.181
  ```
  *(Dado que ya registramos tu clave SSH pública del Mac, se conectará automáticamente de forma segura sin pedirte contraseña).*

### Configuración en VS Code (Mac):
1. Instala la extensión **"Remote - SSH"** (ID: `ms-vscode-remote.remote-ssh`) de Microsoft.
2. Pulsa el botón verde `><` en la parte inferior izquierda de VS Code y selecciona **Connect to Host...** -> **Add New SSH Host...**
3. Introduce: `ssh eudal@192.168.31.181`
4. Selecciona **Windows** como sistema operativo del servidor. Se autenticará automáticamente usando tu clave SSH guardada en tu Mac.

---

## 2. Inferencia de LLMs con Ollama (Servicio en Red)

Ollama ha sido reconfigurado para escuchar peticiones en toda tu red local (interfaz `0.0.0.0`) en lugar de limitarse a localhost.

* **Dirección de la API**: `http://192.168.31.181:11434`
* **Estado del Puerto**: Abierto y verificado (`Ollama is running`)

### Ejemplos para conectar tu Mac:

#### Con la librería oficial `ollama-python`:
```python
import ollama

# Especificamos la IP local de la RTX 4070S
client = ollama.Client(host='http://192.168.31.181:11434')

response = client.chat(model='llama3', messages=[
    {
        'role': 'user',
        'content': '¿Cuál es la distancia entre la Tierra y la Luna?',
    }
])
print(response['message']['content'])
```

#### Con LangChain en Python:
```python
from langchain_community.llms import Ollama

llm = Ollama(
    base_url="http://192.168.31.181:11434",
    model="llama3"  # Asegúrate de que el modelo esté descargado en el PC
)

print(llm.invoke("Hola, responde en español."))
```

---

## 3. Comprobaciones de Red Realizadas
* **Test de Puerto SSH (22)**: Exitoso (`TcpTestSucceeded: True` en `192.168.31.181`).
* **Test de Puerto Ollama (11434)**: Exitoso (Respuesta HTTP 200 recibida en `192.168.31.181:11434`).
