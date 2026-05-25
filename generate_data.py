"""Script para generar datos de telemetría de sensores de servidores con correlaciones físicas realistas.

Simula el comportamiento de uso de CPU, uso de memoria, tráfico de red y temperatura de la CPU,
etiquetando los fallos basándose en sobrecalentamiento y sobrecarga, con ruido añadido.
"""

import os
# Workarounds para evitar problemas de OpenMP en macOS y entornos de cómputo
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# Configuración de logging estructurado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def simulate_server_telemetry(
    num_records: int = 10000,
    server_id: str = "SRV-01",
    seed: int = 42
) -> pd.DataFrame:
    """Genera un DataFrame simulando la telemetría de un servidor con leyes físicas.

    Leyes de simulación:
    1. El uso de CPU y memoria siguen procesos autorregresivos (AR-1) con picos estocásticos.
    2. El tráfico de red está correlacionado positivamente con el uso de CPU.
    3. La temperatura de la CPU sigue la Ley de Enfriamiento de Newton: se calienta por la
       carga de la CPU y se disipa hacia la temperatura ambiente del rack.
    4. Un fallo (Fallo=1) ocurre con un 80% de probabilidad si la temperatura > 85°C y el
       uso de memoria > 90% durante 3 o más pasos consecutivos. Hay un 5% de fallos aleatorios (ruido).

    Args:
        num_records: Número de registros temporales a generar.
        server_id: Identificador único del servidor.
        seed: Semilla aleatoria para reproducibilidad.

    Returns:
        pd.DataFrame con la telemetría simulada del servidor.
    """
    logger.info(f"Iniciando simulación de {num_records} registros para el servidor {server_id}...")
    np.random.seed(seed)
    random.seed(seed)

    # Inicialización de variables de estado y parámetros físicos
    start_time: datetime = datetime(2026, 5, 20, 0, 0, 0)
    timestamps: List[datetime] = [start_time + timedelta(minutes=i) for i in range(num_records)]

    # Parámetros del modelo térmico de la CPU
    temp_ambient: float = 35.0   # Temperatura ambiente dentro del rack (°C)
    temp_cpu: float = 45.0       # Temperatura inicial de la CPU (°C)
    alpha: float = 0.45          # Coeficiente de calentamiento por uso de CPU
    beta: float = 0.05           # Coeficiente de disipación térmica del ventilador

    # Historial de estados para la lógica secuencial
    consecutive_overload_count: int = 0
    records: List[Dict[str, Any]] = []

    # Inicialización de estados autorregresivos
    cpu_usage: float = 20.0
    mem_usage: float = 30.0

    for i in range(num_records):
        # 1. Simulación de Uso de CPU (Proceso AR-1 + picos estocásticos)
        cpu_noise = np.random.normal(0, 3)
        # Evento de pico de carga aleatorio más frecuente para inducir calentamiento
        if random.random() < 0.05:
            cpu_usage = min(98.0, cpu_usage + random.uniform(30, 50))
        else:
            cpu_usage = 0.85 * cpu_usage + 0.15 * 25.0 + cpu_noise
            cpu_usage = max(2.0, min(100.0, cpu_usage))

        # 2. Simulación de Uso de Memoria (Proceso AR-1 lento)
        mem_noise = np.random.normal(0, 1.5)
        if random.random() < 0.04:
            mem_usage = min(99.0, mem_usage + random.uniform(25, 45))
        else:
            # La memoria tiene más inercia que la CPU
            mem_usage = 0.95 * mem_usage + 0.05 * 40.0 + mem_noise
            mem_usage = max(5.0, min(100.0, mem_usage))

        # 3. Tráfico de Red (correlacionado con el uso de CPU + ruido de ráfagas)
        network_traffic = 0.6 * cpu_usage + np.random.exponential(15)
        network_traffic = max(0.1, min(1000.0, network_traffic))  # Limitar en MB/s

        # 4. Simulación de Temperatura de la CPU (Modelo físico dinámico)
        # T_t = T_{t-1} + alpha * (Uso_CPU_t) - beta * (T_{t-1} - T_ambient) + ruido
        temp_noise = np.random.normal(0, 0.5)
        temp_delta_heat = alpha * (cpu_usage / 100.0) * 15.0  # El uso de CPU genera calor
        temp_dissipation = beta * (temp_cpu - temp_ambient)    # Disipación térmica
        temp_cpu = temp_cpu + temp_delta_heat - temp_dissipation + temp_noise
        temp_cpu = max(temp_ambient, min(105.0, temp_cpu))      # Límites físicos de la CPU

        # 5. Determinación lógica del Fallo (Fallo acumulativo de 3 pasos)
        # Umbrales ligeramente más realistas para capturar situaciones de estrés térmico
        is_overloaded = (temp_cpu > 80.0) and (mem_usage > 85.0)
        
        if is_overloaded:
            consecutive_overload_count += 1
        else:
            consecutive_overload_count = 0

        # Lógica de probabilidad de fallo
        is_failure: int = 0
        if consecutive_overload_count >= 3:
            # Alta probabilidad de fallo del sistema por sobrecalentamiento sostenido
            if random.random() < 0.90:
                is_failure = 1
        
        # Ruido de fallo aleatorio (0.1% de probabilidad en cualquier estado)
        if is_failure == 0 and random.random() < 0.001:
            is_failure = 1

        records.append({
            "timestamp": timestamps[i],
            "server_id": server_id,
            "cpu_usage": round(cpu_usage, 2),
            "mem_usage": round(mem_usage, 2),
            "network_traffic": round(network_traffic, 2),
            "cpu_temp": round(temp_cpu, 2),
            "failure": is_failure
        })

    df = pd.DataFrame(records)
    logger.info(f"Simulación finalizada. Dataset con {len(df)} registros generado con éxito.")
    return df


if __name__ == "__main__":
    output_path = "/Users/daldo/VsCode/Tech/dataset_servidores.csv"
    try:
        df_telemetry = simulate_server_telemetry(num_records=10000, seed=42)
        
        # Guardar en archivo CSV
        df_telemetry.to_csv(output_path, index=False)
        logger.info(f"Archivo guardado exitosamente en: {output_path}")
        
        # Estadísticas básicas del dataset
        num_failures = df_telemetry["failure"].sum()
        pct_failures = (num_failures / len(df_telemetry)) * 100
        logger.info(f"Registros totales: {len(df_telemetry)}")
        logger.info(f"Casos de fallo detectados: {num_failures} ({pct_failures:.2f}%)")
    except Exception as e:
        logger.error(f"Error durante la generación de datos: {str(e)}", exc_info=True)
