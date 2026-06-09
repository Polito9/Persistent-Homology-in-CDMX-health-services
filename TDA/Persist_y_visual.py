from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from ripser import ripser

archivo_crudo = Path('matriz_origen_destino.csv') 
archivo_ponderado = Path('matriz_tiempos_ponderada_simetrica.csv')

if archivo_ponderado.is_file():
    # si ya tenemos matriz la matiz simétrica, no calcularla
    df_ponderada = pd.read_csv(archivo_ponderado, index_col=0)
    #Asegurar el mismo tipo de dato en columnas e índices
    df_ponderada.columns = df_ponderada.columns.astype(type(df_ponderada.index[0]))
    matrix_para_tda = df_ponderada.values
    n = df_ponderada.shape[0]
else:
    #en caso de ser la primera vez, leemos el archivo de datos original
    df_raw = pd.read_csv(archivo_crudo)
    
    # segundos a minutos
    df_raw['tiempo_minutos'] = df_raw['tiempo_segundos'] / 60.0
    
    # Identificar todos los nodos únicos
    nodos_totales = sorted(list(set(df_raw['id_origen']).union(set(df_raw['id_destino']))))
    n = len(nodos_totales)

    df_asimetrica = df_raw.pivot(index='id_origen', columns='id_destino', values='tiempo_minutos')
    df_asimetrica = df_asimetrica.reindex(index=nodos_totales, columns=nodos_totales)
    
    #Llenamos vacíos con infinito y extraemos una copia modificable (evita el read-only error) (gemini)
    asym_mat = df_asimetrica.to_numpy(copy=True)
    
    #Modificamos la diagonal con 0 de forma segura
    np.fill_diagonal(asym_mat, 0)
    
    #Simetrización 70-30 inmediata
    max_mat = np.maximum(asym_mat, asym_mat.T)
    min_mat = np.minimum(asym_mat, asym_mat.T)
    #danole el 70% a la más alta y 30% a la más baja
    matrix_para_tda = (0.7 * max_mat) + (0.3 * min_mat)
    
    # Guardamos el resultado final en caché para no volver a hacer esto (gemini)
    df_sym = pd.DataFrame(matrix_para_tda, index=nodos_totales, columns=nodos_totales)
    df_sym.to_csv(archivo_ponderado)

# PROCESAMIENTO GEOMÉTRICO (MDS) (Idea de gimini, modificada)
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


# TDA
#Calculo del diagrama de persistencia
dgms = ripser(matrix_para_tda, distance_matrix=True, maxdim=1)['dgms']
h0_data = dgms[0]
h1_data = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

if len(h1_data) > 0:
    tiempos_muerte_h1 = h1_data[np.isfinite(h1_data[:, 1]), 1]
    max_muerte_h1 = np.max(tiempos_muerte_h1) if len(tiempos_muerte_h1) > 0 else np.max(matrix_para_mds)
else:
    max_muerte_h1 = np.max(matrix_para_mds)

limite_vista = max_muerte_h1 * 1.05


# CONFIGURACIÓN DE LA GRÁFICA PARA ANIMACIÓN (gemini)
fig, (ax_mds, ax_pers) = plt.subplots(1, 2, figsize=(15, 7))
plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.3)

ax_mds.scatter(points[:, 0], points[:, 1], color='black', s=2, zorder=3, alpha=0.6)
ax_mds.set_aspect('equal')
ax_mds.set_xlim(np.min(points[:, 0]) - 5, np.max(points[:, 0]) + 5)
ax_mds.set_ylim(np.min(points[:, 1]) - 5, np.max(points[:, 1]) + 5)

# Colección optimizada para las conexiones rojas
edge_collection = LineCollection([], colors='red', linewidths=0.5, alpha=0.15, zorder=1)
ax_mds.add_collection(edge_collection)

ax_pers.plot([0, limite_vista], [0, limite_vista], color='gray', linestyle='--', alpha=0.5)
ax_pers.set_xlabel("Tiempo de Nacimiento (minutos)")
ax_pers.set_ylabel("Tiempo de Muerte (minutos)")
ax_pers.set_xlim(-1, limite_vista)
ax_pers.set_ylim(-1, limite_vista)

scat_h0 = ax_pers.scatter([], [], color='orange', s=20, label='H0 (Registro)', alpha=0.8)
scat_h1 = ax_pers.scatter([], [], color='purple', s=35, label='H1 (Registro)', marker='^', alpha=0.9)
ax_pers.legend(loc='lower right')


# ANIMACIÓN DINÁMICA (gemini)
num_frames = 60
valores_epsilon = np.linspace(0.0, limite_vista, num_frames)

def animate(frame):
    epsilon = valores_epsilon[frame]
    
    # 1. Actualizar Grafo (Panel Izquierdo)
    ax_mds.set_title(f"Complejo Simplicial\nRadio = {epsilon:.2f} min", fontsize=11, fontweight='bold')
    
    ij_conexiones = np.argwhere((matrix_para_tda <= 2 * epsilon) & (matrix_para_tda > 0))
    if len(ij_conexiones) > 0:
        ij_conexiones = ij_conexiones[ij_conexiones[:, 0] < ij_conexiones[:, 1]]
        p_start = points[ij_conexiones[:, 0]]
        p_end = points[ij_conexiones[:, 1]]
        
        # Muestreo para no saturar la memoria gráfica con millones de líneas
        if len(p_start) > 5000:
            indices_muestra = np.random.choice(len(p_start), 5000, replace=False)
            p_start = p_start[indices_muestra]
            p_end = p_end[indices_muestra]

        # Actualizamos la colección en un solo golpe de memoria
        segments = np.stack((p_start, p_end), axis=1)
        edge_collection.set_segments(segments)
    else:
        edge_collection.set_segments([])

    # 2. Actualizar Contadores Directos H0 y H1 (Panel Derecho)
    vivos_h0 = h0_data[(h0_data[:, 0] <= epsilon) & (h0_data[:, 1] > epsilon)]
    vivos_h1 = h1_data[(h1_data[:, 0] <= epsilon) & (h1_data[:, 1] > epsilon)] if len(h1_data) > 0 else []
    
    ax_pers.set_title(f"Evolución Topológica\nH0 (Componentes): {len(vivos_h0)} | H1 (Huecos): {len(vivos_h1)}", 
                     fontsize=11, fontweight='bold')

    valid_h0 = h0_data[h0_data[:, 0] <= epsilon]
    if len(valid_h0) > 0:
        valid_h0_vis = valid_h0.copy()
        valid_h0_vis[~np.isfinite(valid_h0_vis)] = limite_vista * 0.95
        scat_h0.set_offsets(valid_h0_vis)
    else:
        scat_h0.set_offsets(np.empty((0, 2)))

    valid_h1 = h1_data[h1_data[:, 0] <= epsilon] if len(h1_data) > 0 else []
    if len(valid_h1) > 0:
        scat_h1.set_offsets(valid_h1)
    else:
        scat_h1.set_offsets(np.empty((0, 2)))
        
    return edge_collection, scat_h0, scat_h1


# EXPORTACIÓN 
ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=350, blit=True)
nombre_gif = 'filtracion_tda_real_completa.gif'
ani.save(nombre_gif, writer='pillow', fps=3)