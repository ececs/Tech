"""
Template: Pipeline de Entrenamiento en PyTorch
Este script contiene una estructura estándar de producción para entrenar modelos de Deep Learning:
1. Dataset personalizado y DataLoader.
2. Arquitectura de Red Neuronal (ejemplo modular).
3. Bucle de Entrenamiento y Validación (Training Loop con early stopping y cálculo de métricas).
4. Guardado de checkpoints y visualización de curvas de pérdida/precisión.
"""

import os
import sys

# Workarounds de OpenMP para evitar segfaults en macOS con PyTorch/FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# ==========================================
# 1. CONFIGURACIÓN DE DISPOSITIVO (¡Compatible con Apple Silicon/MPS!)
# ==========================================
device = torch.device(
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
)
print(f"[+] Usando dispositivo de cómputo: {device}")


# ==========================================
# 2. DATASET PERSONALIZADO
# ==========================================
class CustomDataset(Dataset):
    """
    Reemplaza esta clase con la lógica de tu dataset.
    Ejemplo: cargar imágenes, tokens de texto o vectores de características.
    """
    def __init__(self, num_samples: int = 1000, num_features: int = 20, num_classes: int = 2):
        # Datos simulados para demostración
        self.x_data = torch.randn(num_samples, num_features, dtype=torch.float32)
        # Clases aleatorias para clasificación binaria/multiclase
        self.y_data = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x_data[idx], self.y_data[idx]


# ==========================================
# 3. ARQUITECTURA DE LA RED NEURONAL
# ==========================================
class ClassificationNet(nn.Module):
    """
    Una red neuronal feed-forward (MLP) estándar con Dropout y Batch Normalization.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super(ClassificationNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ==========================================
# 4. BUCLE DE ENTRENAMIENTO Y VALIDACIÓN
# ==========================================
class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        criterion: nn.modules.loss._Loss, 
        optimizer: optim.Optimizer,
        device: torch.device
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        # Historial para graficar
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass & Optimize
            loss.backward()
            self.optimizer.step()
            
            # Métricas
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, patience: int = 5):
        best_val_loss = float("inf")
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(
                f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%"
            )
            
            # Early Stopping y Guardado del mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), "best_model.pth")
                print("  [+] Guardado nuevo checkpoint (mejor pérdida de validación).")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\n[!] Early stopping activado tras {patience} épocas sin mejora.")
                    break

    def plot_metrics(self):
        """Grafica y guarda las curvas de pérdida y precisión."""
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Val Loss")
        plt.title("Curva de Pérdida (Loss)")
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accs, label="Train Acc")
        plt.plot(epochs, self.val_accs, label="Val Acc")
        plt.title("Curva de Precisión (Accuracy)")
        plt.xlabel("Épocas")
        plt.ylabel("Precisión")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("training_curves.png")
        print("[+] Gráficas de entrenamiento guardadas como 'training_curves.png'")
        plt.show()


# ==========================================
# 5. EJECUCIÓN DEL PIPELINE
# ==========================================
if __name__ == "__main__":
    # Hyperparámetros
    EPOCHS = 20
    BATCH_SIZE = 32
    LR = 0.001
    INPUT_DIM = 20
    HIDDEN_DIM = 64
    NUM_CLASSES = 2
    
    # Crear datasets y loaders
    train_dataset = CustomDataset(num_samples=800, num_features=INPUT_DIM, num_classes=NUM_CLASSES)
    val_dataset = CustomDataset(num_samples=200, num_features=INPUT_DIM, num_classes=NUM_CLASSES)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Inicializar red, criterio y optimizador
    net = ClassificationNet(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4) # weight_decay es L2 regularization
    
    # Instanciar trainer y entrenar
    trainer = Trainer(model=net, criterion=loss_fn, optimizer=opt, device=device)
    print("\n[*] Iniciando ciclo de entrenamiento...")
    trainer.fit(train_loader, val_loader, epochs=EPOCHS, patience=5)
    
    # Graficar resultados
    # Nota: Si estás ejecutándolo en consola sin entorno visual, plot_metrics guardará la imagen correctamente.
    try:
        trainer.plot_metrics()
    except Exception as e:
        print(f"[!] No se pudo graficar: {e}")
