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
    asym_mat = df_asimetrica.fillna(np.inf).to_numpy(copy=True)
    
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
# Calculo del diagrama de persistencia
dgms = ripser(matrix_para_tda, distance_matrix=True, maxdim=1)['dgms']
h0_data = dgms[0]
h1_data = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

def filtrar_por_desviacion(datos, desviaciones=1.0):
    """Calcula la persistencia y separa la topología en 'significativa' y 'ruido'."""
    if len(datos) == 0:
        return datos, np.empty((0, 2)), 0.0
        
    # Calcular persistencia solo para puntos finitos (evita que el Infinito de H0 rompa la estadística)
    finitos = datos[np.isfinite(datos[:, 1])]
    if len(finitos) == 0:
        return datos, np.empty((0, 2)), 0.0
        
    persistencia = finitos[:, 1] - finitos[:, 0]
    media = np.mean(persistencia)
    desviacion_std = np.std(persistencia)
    
    umbral = media + (desviaciones * desviacion_std)
    
    # Evaluar todos los puntos (manejando advertencias de infinito de forma segura)
    with np.errstate(invalid='ignore'):
        persistencia_total = datos[:, 1] - datos[:, 0]
    
    es_infinito = np.isinf(datos[:, 1])
    # Es significativo si supera el umbral o si es la componente principal (infinita)
    es_significativo = (persistencia_total >= umbral) | es_infinito
    
    significativos = datos[es_significativo]
    ruido = datos[~es_significativo]
    
    return significativos, ruido, umbral

# Parámetro de sensibilidad: 1.0 a 1.5 es lo estándar
num_std = 2.0 
h0_sig, h0_ruido, umbral_h0 = filtrar_por_desviacion(h0_data, desviaciones=num_std)
h1_sig, h1_ruido, umbral_h1 = filtrar_por_desviacion(h1_data, desviaciones=num_std)

# Impresión de las listas solicitadas en consola
print(f"--- ANÁLISIS DE PERSISTENCIA (Umbral: {num_std} Desviaciones Estándar) ---")
print(f"H0 Significativos encontrados: {len(h0_sig)} (Umbral de vida: > {umbral_h0:.2f} min)")
print(f"H1 Significativos encontrados: {len(h1_sig)} (Umbral de vida: > {umbral_h1:.2f} min)")

# Preparar límites de vista para la gráfica
if len(h1_data) > 0:
    tiempos_muerte_h1 = h1_data[np.isfinite(h1_data[:, 1]), 1]
    max_muerte_h1 = np.max(tiempos_muerte_h1) if len(tiempos_muerte_h1) > 0 else np.max(matrix_para_mds)
else:
    max_muerte_h1 = np.max(matrix_para_mds)

limite_vista = max_muerte_h1 * 1.05


# CONFIGURACIÓN DE LA GRÁFICA PARA ANIMACIÓN
fig, (ax_mds, ax_pers) = plt.subplots(1, 2, figsize=(15, 7))
plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.3)

# Panel Izquierdo: MDS
ax_mds.scatter(points[:, 0], points[:, 1], color='black', s=2, zorder=3, alpha=0.6)
ax_mds.set_aspect('equal')
ax_mds.set_xlim(np.min(points[:, 0]) - 5, np.max(points[:, 0]) + 5)
ax_mds.set_ylim(np.min(points[:, 1]) - 5, np.max(points[:, 1]) + 5)

edge_collection = LineCollection([], colors='red', linewidths=0.5, alpha=0.15, zorder=1)
ax_mds.add_collection(edge_collection)

# Panel Derecho: Diagrama de Persistencia
ax_pers.plot([0, limite_vista], [0, limite_vista], color='gray', linestyle='--', alpha=0.5)
ax_pers.set_xlabel("Tiempo de Nacimiento (minutos)")
ax_pers.set_ylabel("Tiempo de Muerte (minutos)")
ax_pers.set_xlim(-1, limite_vista)
ax_pers.set_ylim(-1, limite_vista)

# Inicializamos el scatter del RUIDO (Pequeño y gris/transparente)
scat_h0_ruido = ax_pers.scatter([], [], color='gray', s=10, alpha=0.3)
scat_h1_ruido = ax_pers.scatter([], [], color='gray', s=15, marker='^', alpha=0.3)

# Inicializamos el scatter SIGNIFICATIVO (Grande, brillante y con borde)
scat_h0_sig = ax_pers.scatter([], [], color='darkorange', s=50, label='H0 Significativo', alpha=0.9, edgecolor='black')
scat_h1_sig = ax_pers.scatter([], [], color='purple', s=70, label='H1 Significativo', marker='^', alpha=0.9, edgecolor='black')

ax_pers.legend(loc='lower right')


# ANIMACIÓN DINÁMICA
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
        
        if len(p_start) > 5000:
            indices_muestra = np.random.choice(len(p_start), 5000, replace=False)
            p_start = p_start[indices_muestra]
            p_end = p_end[indices_muestra]

        segments = np.stack((p_start, p_end), axis=1)
        edge_collection.set_segments(segments)
    else:
        edge_collection.set_segments([])

    # 2. Actualizar Diagrama de Persistencia (Panel Derecho)
    # Contamos cuántos de los SIGNIFICATIVOS están vivos en este epsilon
    vivos_h0_sig = h0_sig[(h0_sig[:, 0] <= epsilon) & (h0_sig[:, 1] > epsilon)]
    vivos_h1_sig = h1_sig[(h1_sig[:, 0] <= epsilon) & (h1_sig[:, 1] > epsilon)] if len(h1_sig) > 0 else []
    
    ax_pers.set_title(f"Topología Significativa (>{num_std} std)\nH0 Activos: {len(vivos_h0_sig)} | H1 Activos: {len(vivos_h1_sig)}", 
                      fontsize=11, fontweight='bold', color='darkred')

    # Función auxiliar para actualizar scatters manejando infinitos
    def actualizar_scatter(scatter_obj, datos_completos):
        validos = datos_completos[datos_completos[:, 0] <= epsilon]
        if len(validos) > 0:
            vis = validos.copy()
            vis[~np.isfinite(vis)] = limite_vista * 0.95 # Reemplazar inf visualmente
            scatter_obj.set_offsets(vis)
        else:
            scatter_obj.set_offsets(np.empty((0, 2)))

    # Actualizamos las 4 categorías
    actualizar_scatter(scat_h0_ruido, h0_ruido)
    actualizar_scatter(scat_h1_ruido, h1_ruido)
    actualizar_scatter(scat_h0_sig, h0_sig)
    actualizar_scatter(scat_h1_sig, h1_sig)
        
    return edge_collection, scat_h0_ruido, scat_h1_ruido, scat_h0_sig, scat_h1_sig


# EXPORTACIÓN 
ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=350, blit=True)
nombre_gif = 'filtracion_tda_significativa.gif'
ani.save(nombre_gif, writer='pillow', fps=3)