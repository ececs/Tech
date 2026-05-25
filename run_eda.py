"""Script para ejecutar el Análisis Exploratorio de Datos (EDA) y generar las imágenes de visualización.
"""

import os
# Workarounds para evitar problemas de OpenMP en macOS y entornos de cómputo
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de logging estructurado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_eda(df_path: str = "/Users/daldo/VsCode/Tech/dataset_servidores.csv") -> None:
    """Ejecuta el EDA y guarda las gráficas en disco."""
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"No se encontró el dataset en: {df_path}")

    logger.info(f"Cargando dataset desde: {df_path}")
    df = pd.read_csv(df_path)

    # 1. Distribución de clases
    logger.info("Generando distribución de clases...")
    class_counts = df['failure'].value_counts()
    class_percentages = df['failure'].value_counts(normalize=True) * 100
    
    logger.info(f"Normal: {class_counts.get(0, 0)} ({class_percentages.get(0, 0.0):.2f}%)")
    logger.info(f"Fallo: {class_counts.get(1, 0)} ({class_percentages.get(1, 0.0):.2f}%)")

    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, legend=False, palette='viridis')
    plt.title('Distribución de Clases (Fallo vs Normal)')
    plt.xlabel('Fallo (0 = Normal, 1 = Fallo)')
    plt.ylabel('Cantidad de Registros')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('/Users/daldo/VsCode/Tech/distribution_clases.png', bbox_inches='tight')
    plt.close()

    # 2. Matriz de correlación
    logger.info("Generando matriz de correlación...")
    numeric_cols = ['cpu_usage', 'mem_usage', 'network_traffic', 'cpu_temp', 'failure']
    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matriz de Correlación de Pearson')
    plt.savefig('/Users/daldo/VsCode/Tech/matriz_correlacion.png', bbox_inches='tight')
    plt.close()

    # 3. Distribución de variables por clase de fallo
    logger.info("Generando distribución de variables clave por estado...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma de Temperatura de la CPU según Fallo
    sns.histplot(data=df, x='cpu_temp', hue='failure', kde=True, ax=axes[0], palette='Set1', bins=40, alpha=0.6)
    axes[0].set_title('Distribución de Temperatura de CPU por Estado')
    axes[0].set_xlabel('Temperatura CPU (°C)')
    axes[0].set_ylabel('Frecuencia')

    # Boxplot de Uso de Memoria según Fallo
    sns.boxplot(data=df, x='failure', y='mem_usage', ax=axes[1], hue='failure', legend=False, palette='Set2')
    axes[1].set_title('Uso de Memoria por Estado')
    axes[1].set_xlabel('Estado (0 = Normal, 1 = Fallo)')
    axes[1].set_ylabel('Uso Memoria (%)')

    plt.savefig('/Users/daldo/VsCode/Tech/distribucion_variables_fallo.png', bbox_inches='tight')
    plt.close()
    
    logger.info("EDA completado y gráficos guardados con éxito.")


if __name__ == "__main__":
    try:
        run_eda()
    except Exception as e:
        logger.error(f"Error durante el EDA: {str(e)}", exc_info=True)
