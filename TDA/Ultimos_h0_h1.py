from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from ripser import ripser

# --- 1. CARGAR MATRIZ DESDE EL CACHÉ ---
archivo_ponderado = Path('matriz_tiempos_ponderada_simetrica.csv')

if not archivo_ponderado.is_file():
    raise FileNotFoundError(f"No se encontró '{archivo_ponderado}'. Ejecuta primero el código anterior para generar el caché.")

print(f"-> Cargando matriz simétrica desde '{archivo_ponderado}'...")
df_ponderada = pd.read_csv(archivo_ponderado, index_col=0)
matrix_para_tda = df_ponderada.values
n = df_ponderada.shape[0]

# --- 2. MDS (PROYECTAR A ESPACIO 2D) ---
print("-> Calculando MDS...")
matrix_para_mds = matrix_para_tda.copy()
if np.isinf(matrix_para_mds).any():
    max_finito = np.max(matrix_para_mds[~np.isinf(matrix_para_mds)])
    matrix_para_mds[np.isinf(matrix_para_mds)] = max_finito * 2

H = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * H @ (matrix_para_mds ** 2) @ H
evals, evecs = np.linalg.eigh(B)
idx = np.argsort(evals)[::-1]
evals_top = np.maximum(evals[idx[:2]], 0)
points = evecs[:, idx[:2]] * np.sqrt(evals_top)

# --- 3. CALCULAR RIPSER Y ORDENAR H1 POR PERSISTENCIA ---
print("-> Calculando Persistencia con Ripser...")
dgms = ripser(matrix_para_tda, distance_matrix=True, maxdim=1)['dgms']
h1_data = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

if len(h1_data) == 0:
    print("¡Alerta! No se detectaron huecos H1 en los datos.")
    exit()

# Calcular la persistencia de cada hueco (Muerte - Nacimiento)
# Para los que tienen muerte infinita, usamos el máximo finito como referencia
h1_limpio = h1_data.copy()
h1_limpio[~np.isfinite(h1_limpio[:, 1]), 1] = np.max(matrix_para_mds)
persistencias = h1_limpio[:, 1] - h1_limpio[:, 0]

# Ordenar de mayor a menor persistencia
indices_ordenados = np.argsort(persistencias)[::-1]

print("\n========================================================")
print("     REPORTE TOPOLÓGICO: TOP 11 HUECOS MÁS PERSISTENTES")
print("========================================================")
for i, idx_h1 in enumerate(indices_ordenados[:11]):
    nacimiento = h1_limpio[idx_h1, 0]
    muerte = h1_limpio[idx_h1, 1]
    pers = persistencias[idx_h1]
    
    categoria = "★ TOP 2 MASTER" if i < 2 else "Top 11 Miembro"
    print(f"[{categoria}] Rank {i+1:2d} | Nace: {nacimiento:5.2f} min | Muere: {muerte:5.2f} min | Persistencia: {pers:5.2f} min")

# --- 4. SELECCIÓN DE RADIOS CRÍTICOS PARA LA VISUALIZACIÓN ---
# Vamos a definir 4 radios de filtración (epsilon) basados en los resultados:
radio_inicial = h1_limpio[indices_ordenados[10], 0]  # Nacimiento del Rank 11
radio_top11_vivo = h1_limpio[indices_ordenados[10], 0] + 1.0  # Momento donde los 11 están vivos
radio_top2_solo = h1_limpio[indices_ordenados[2], 1] + 1.0    # Justo después de que muere el Rank 3 (solo quedan los 2 masters)
radio_final = h1_limpio[indices_ordenados[0], 1] - 0.5        # Justo antes de que muera el hueco más grande de todos

radios_criticos = [radio_inicial, radio_top11_vivo, radio_top2_solo, radio_final]
titulos = [
    f"A) Epsilon = {radio_inicial:.2f} min\n(Nace el Rank 11)",
    f"B) Epsilon = {radio_top11_vivo:.2f} min\n(Los 11 macro-bucles están activos)",
    f"C) Epsilon = {radio_top2_solo:.2f} min\n(Murieron los demás, quedan los 2 principales)",
    f"D) Epsilon = {radio_final:.2f} min\n(Último aliento del bucle estructural definitivo)"
]

# --- 5. GRAFICAR LAS 4 INSTANTÁNEAS DE LA FILTRACIÓN ---
print("\n-> Generando gráfica comparativa de radios críticos...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for k, epsilon in enumerate(radios_criticos):
    ax = axes[k]
    # Dibujar los puntos base
    ax.scatter(points[:, 0], points[:, 1], color='black', s=3, zorder=3, alpha=0.5)
    
    # Encontrar qué conexiones existen a este radio específico (Rips: d <= 2*epsilon)
    ij_conexiones = np.argwhere((matrix_para_tda <= 2 * epsilon) & (matrix_para_tda > 0))
    
    if len(ij_conexiones) > 0:
        # Evitar duplicados (i,j) y (j,i)
        ij_conexiones = ij_conexiones[ij_conexiones[:, 0] < ij_conexiones[:, 1]]
        p_start = points[ij_conexiones[:, 0]]
        p_end = points[ij_conexiones[:, 1]]
        
        # Muestreo adaptativo para que la gráfica no sea un manchón negro ilegible
        if len(p_start) > 8000:
            indices_muestra = np.random.choice(len(p_start), 8000, replace=False)
            p_start = p_start[indices_muestra]
            p_end = p_end[indices_muestra]
            
        segments = np.stack((p_start, p_end), axis=1)
        # Usamos LineCollection para pintar rápido y con transparencia
        edge_collection = LineCollection(segments, colors='red', linewidths=0.4, alpha=0.1, zorder=1)
        ax.add_collection(edge_collection)
        
    ax.set_title(titulos[k], fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(np.min(points[:, 0]) - 5, np.max(points[:, 0]) + 5)
    ax.set_ylim(np.min(points[:, 1]) - 5, np.max(points[:, 1]) + 5)
    ax.set_xlabel("Dimensión MDS 1")
    ax.set_ylabel("Dimensión MDS 2")

plt.suptitle("Evolución Geométrica de los Macro-Bucles de Conectividad (H1)", fontsize=16, fontweight='bold', y=0.96)
plt.subplots_adjust(hspace=0.25, wspace=0.25)

nombre_salida = 'analisis_radios_criticos_h1.png'
plt.savefig(nombre_salida, dpi=200, bbox_inches='tight')
print(f"¡Listo! Gráfica guardada como '{nombre_salida}'.")