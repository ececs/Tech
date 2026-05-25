# Guía: Configurar tu PC (RTX 4070 Super) como Servidor de Cómputo para tu Mac

Para evitar la molestia de tener que cambiar físicamente de ordenador, puedes configurar tu PC secundario (con Windows/Linux y la RTX 4070 Super) como un servidor de cómputo remoto. De esta manera, **escribirás y controlarás todo desde tu MacBook Air M4**, pero el procesamiento pesado y la inferencia de modelos se ejecutarán en la tarjeta gráfica del PC.

Aquí tienes las 3 formas más eficientes de hacerlo, de menor a mayor complejidad.

---

## Método 1: Exponer Ollama en red local (Para Inferencia de LLMs locales en la GPU)

Si deseas ejecutar un modelo de lenguaje local (como `llama3:8b` o `phi3` en la RTX 4070S) pero hacer las consultas RAG desde tu script de Python en el Mac:

### En el PC (Servidor):
1. **Configurar la variable de entorno** para permitir conexiones externas:
   * **En Windows (Powershell)**: 
     ```powershell
     [Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0", "User")
     ```
     *(Reinicia Ollama tras aplicar esto).*
   * **En Linux**: Edita el servicio de systemd de Ollama y añade `Environment="OLLAMA_HOST=0.0.0.0"`.
2. Busca la IP local de tu PC en la red wifi/ethernet (ej. `192.168.1.50`).

### En el Mac (Cliente):
En tu código de Python (con LangChain u Ollama), apunta el cliente a la dirección IP del PC en lugar de `localhost`:

```python
# Con LangChain
from langchain_community.llms import Ollama

llm = Ollama(
    base_url="http://192.168.1.50:11434",  # Reemplaza con la IP real de tu PC
    model="llama3:8b"
)
```

---

## Método 2: VS Code Remote - SSH (Para programar y entrenar directamente en la GPU)

Este es el estándar profesional. Trabajarás en la ventana de VS Code de tu Mac, pero todos los comandos de la terminal y ejecuciones de código de Python se harán físicamente en el PC con soporte CUDA.

### Paso 1: Configurar SSH en el PC
* **Si tu PC tiene Windows**:
  1. Ve a Configuración > Aplicaciones > Características opcionales.
  2. Asegúrate de tener instalado "Servidor OpenSSH". Si no, agrégalo.
  3. Abre Powershell como administrador y activa el servicio:
     ```powershell
     Start-Service sshd
     Set-Service -Name sshd -StartupType 'Automatic'
     ```
* **Si tu PC tiene Linux**:
  ```bash
  sudo apt install openssh-server
  sudo systemctl enable --now ssh
  ```

### Paso 2: Conectar desde el MacBook Air
1. Instala la extensión **"Remote - SSH"** (de Microsoft) en el VS Code de tu Mac.
2. Haz clic en el botón verde de la esquina inferior izquierda de VS Code (`><`) y selecciona **Connect to Host... > Add New SSH Host**.
3. Escribe el comando de conexión: `ssh usuario_del_pc@IP_DEL_PC` (ej. `ssh devuser@192.168.1.50`).
4. Introduce la contraseña del PC cuando la solicite.
5. ¡Listo! Abre la carpeta de tu proyecto en el PC desde esa misma ventana. La terminal integrada de VS Code ejecutará todo directamente en el PC usando CUDA.

---

## Método 3: Servidor de Jupyter remoto (Para entrenamiento interactivo en PyTorch)

Si quieres hacer prototipos o entrenamientos interactivos en formato Notebook pero ejecutándolos en la RTX 4070 Super:

### En el PC:
1. Abre tu terminal con el entorno virtual activado y ejecuta:
   ```bash
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
   ```
2. La terminal mostrará un enlace con un token de acceso:
   `http://127.0.0.1:8888/lab?token=abcdef123456...`

### En el Mac:
1. Abre el navegador web en tu Mac.
2. Reemplaza `127.0.0.1` por la IP local del PC en el enlace que te dio la terminal:
   `http://192.168.1.50:8888/lab?token=abcdef123456...`
3. Ahora podrás crear celdas de PyTorch y correr bucles de entrenamiento pesado utilizando los 12GB de VRAM de la RTX 4070 Super directamente desde tu navegador del MacBook Air.
