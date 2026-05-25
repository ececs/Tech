"""Script para verificar si existe desplazamiento de datos (Data Drift) entre los splits temporales.
"""

import os
# Workarounds para evitar problemas de OpenMP en macOS y entornos de cómputo
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def analyze_splits(df_path: str = "/Users/daldo/VsCode/Tech/dataset_servidores.csv") -> None:
    """Divide el dataset secuencialmente en 70/15/15 y compara las estadísticas descriptivas."""
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"No se encontró el dataset en {df_path}")
        
    df = pd.read_csv(df_path)
    n = len(df)
    
    # Splits secuenciales
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    logger.info("=== ANÁLISIS DE SPLITS TEMPORALES (70 / 15 / 15) ===")
    logger.info(f"Registros - Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    features = ["cpu_usage", "mem_usage", "network_traffic", "cpu_temp"]
    
    # Tabla comparativa de medias y desv. estándar
    summary_data = []
    for feat in features:
        summary_data.append({
            "Feature": feat,
            "Train Mean": train_df[feat].mean(),
            "Train Std": train_df[feat].std(),
            "Val Mean": val_df[feat].mean(),
            "Val Std": val_df[feat].std(),
            "Test Mean": test_df[feat].mean(),
            "Test Std": test_df[feat].std(),
        })
        
    summary_df = pd.DataFrame(summary_data)
    print("\nEstadísticas de Características:")
    print(summary_df.to_string(index=False, formatters={
        "Train Mean": "{:.2f}".format, "Train Std": "{:.2f}".format,
        "Val Mean": "{:.2f}".format, "Val Std": "{:.2f}".format,
        "Test Mean": "{:.2f}".format, "Test Std": "{:.2f}".format
    }))
    
    # Análisis del target (fallos)
    print("\nProporción de Anomalías (Fallo = 1):")
    train_fail = train_df["failure"].sum()
    val_fail = val_df["failure"].sum()
    test_fail = test_df["failure"].sum()
    
    print(f"Train Failures: {train_fail} ({train_fail/len(train_df)*100:.2f}%)")
    print(f"Val Failures:   {val_fail} ({val_fail/len(val_df)*100:.2f}%)")
    print(f"Test Failures:  {test_fail} ({test_fail/len(test_df)*100:.2f}%)")


if __name__ == "__main__":
    analyze_splits()
