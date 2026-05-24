import os
import sys

# Prevenir fallos por duplicado de OpenMP en macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


print("=== DIAGNÓSTICO DEL ENTORNO DE IA/ML ===")

print(f"Versión de Python: {sys.version}\n")

# 1. Verificar PyTorch y MPS (Apple Silicon GPU)
try:
    import torch
    print(f"[✓] PyTorch instalado. Versión: {torch.__version__}")
    
    # Comprobar si MPS (Metal Performance Shaders) está disponible
    mps_available = torch.backends.mps.is_available()
    print(f"    - ¿Apple Silicon GPU (MPS) disponible?: {mps_available}")
    if mps_available:
        device = torch.device("mps")
        # Hacer una prueba de tensor simple en GPU de Mac
        x = torch.ones(5, device=device)
        y = x * 2
        print(f"    - Prueba de tensor en MPS exitosa: {y.cpu().numpy()}")
    else:
        print("    - [!] MPS no disponible. Se usará CPU por defecto si no hay GPU CUDA.")
except ImportError:
    print("[X] PyTorch NO está instalado.")
except Exception as e:
    print(f"[X] Error probando PyTorch/MPS: {e}")

print()

# 2. Verificar NumPy, Pandas y Matplotlib
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    print(f"[✓] NumPy instalado. Versión: {np.__version__}")
    print(f"[✓] Pandas instalado. Versión: {pd.__version__}")
    print(f"[✓] Matplotlib instalado. Versión: {matplotlib.__version__}")
except ImportError as e:
    print(f"[X] Error importando librerías Core: {e}")

print()

# 3. Verificar Hugging Face Transformers
try:
    import transformers
    import datasets
    import accelerate
    print(f"[✓] Transformers instalado. Versión: {transformers.__version__}")
    print(f"[✓] Datasets instalado. Versión: {datasets.__version__}")
    print(f"[✓] Accelerate instalado. Versión: {accelerate.__version__}")
except ImportError as e:
    print(f"[X] Error importando Hugging Face suite: {e}")

print()

# 4. Verificar FAISS y LangChain
try:
    import faiss
    import langchain
    print(f"[✓] FAISS instalado. Versión: {faiss.__version__}")
    print(f"[✓] LangChain instalado. Versión: {langchain.__version__}")
    
    # Comprobar indexación simple en FAISS
    dimension = 64
    nb = 1000
    nq = 5
    np.random.seed(1234)
    xb = np.random.random((nb, dimension)).astype('float32')
    xq = np.random.random((nq, dimension)).astype('float32')
    
    index = faiss.IndexFlatL2(dimension)
    index.add(xb)
    D, I = index.search(xq, 4)
    print(f"    - Prueba de indexación en FAISS exitosa. Top 1 index: {I[0][0]}")
except ImportError as e:
    print(f"[X] Error importando FAISS o LangChain: {e}")
except Exception as e:
    print(f"[X] Error probando FAISS: {e}")

print("\n========================================")
