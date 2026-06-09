from pathlib import Path
import contextily as ctx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
from ripser import ripser
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

# --- 1. CARGA O CÁLCULO DE LA MATRIZ SIMÉTRICA ---
archivo_crudo = Path('matriz_origen_destino.csv') 
archivo_ponderado = Path('matriz_tiempos_ponderada_simetrica.csv')

if archivo_ponderado.is_file():
    print("-> Cargando matriz simétrica desde caché...")
    df_ponderada = pd.read_csv(archivo_ponderado, index_col=0)
    df_ponderada.columns = df_ponderada.columns.astype(type(df_ponderada.index[0]))
    matrix_para_tda = df_ponderada.values
    n = df_ponderada.shape[0]
    nodos_totales = df_ponderada.index.tolist()
else:
    print("-> Procesando matriz cruda...")
    df_raw = pd.read_csv(archivo_crudo)
    df_raw['tiempo_minutos'] = df_raw['tiempo_segundos'] / 60.0
    nodos_totales = sorted(list(set(df_raw['id_origen']).union(set(df_raw['id_destino']))))
    n = len(nodos_totales)

    df_asimetrica = df_raw.pivot(index='id_origen', columns='id_destino', values='tiempo_minutos')
    df_asimetrica = df_asimetrica.reindex(index=nodos_totales, columns=nodos_totales)
    
    asym_mat = df_asimetrica.to_numpy(copy=True)
    np.fill_diagonal(asym_mat, 0)
    
    max_mat = np.maximum(asym_mat, asym_mat.T)
    min_mat = np.minimum(asym_mat, asym_mat.T)
    matrix_para_tda = (0.7 * max_mat) + (0.3 * min_mat)
    
    df_sym = pd.DataFrame(matrix_para_tda, index=nodos_totales, columns=nodos_totales)
    df_sym.to_csv(archivo_ponderado)

# --- 2. TDA Y ESTRUCTURA JERÁRQUICA (H0) ---
print("-> Ejecutando Ripser y construyendo enlaces jerárquicos...")
dgms = ripser(matrix_para_tda, distance_matrix=True, maxdim=0)['dgms']
h0_data = dgms[0]

matrix_dendro = matrix_para_tda.copy()
if np.isinf(matrix_dendro).any():
    max_finito = np.max(matrix_dendro[~np.isinf(matrix_dendro)])
    matrix_dendro[np.isinf(matrix_dendro)] = max_finito * 2
np.fill_diagonal(matrix_dendro, 0)
matrix_dendro = np.maximum(matrix_dendro, matrix_dendro.T)

distancias_condensadas = ssd.squareform(matrix_dendro)
Z = sch.linkage(distancias_condensadas, method='single')

# --- 3. PROCESAMIENTO GEOMÉTRICO (MDS + COORDENADAS GEOGRÁFICAS) ---
print("-> Calculando espacio topológico (MDS)...")
H = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * H @ (matrix_dendro ** 2) @ H
evals, evecs = np.linalg.eigh(B)
idx = np.argsort(evals)[::-1]
evals_top = np.maximum(evals[idx[:2]], 0)
points_mds = evecs[:, idx[:2]] * np.sqrt(evals_top)

print("-> Alineando coordenadas geográficas reales...")
df = pd.read_csv('denue_inegi_62_.csv', encoding="latin-1")
filter_data = df[df["cve_ent"].isin([9])]
codigos = [621111, 621112, 621115, 621116, 621491, 621492, 622111, 622112]
df_coords = filter_data[filter_data['codigo_act'].isin(codigos)].set_index('id')
df_coords = df_coords.reindex(nodos_totales)

lats = df_coords['latitud'].values
lons = df_coords['longitud'].values
points_geo = np.column_stack((lons, lats))

# --- 4. DETERMINACIÓN DEL LÍMITE DEL SLIDER ---
limite_slider = 6.0

# --- 5. CONFIGURACIÓN DE LA INTERFAZ GRÁFICA INTERACTIVA (LADO A LADO) ---
fig, (ax_mds, ax_geo) = plt.subplots(1, 2, figsize=(18, 9))
plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.18, wspace=0.12)

labels_iniciales = np.arange(n)

# A) Panel Izquierdo: Nube MDS
scatter_mds = ax_mds.scatter(points_mds[:, 0], points_mds[:, 1], c=labels_iniciales, 
                             cmap='turbo', s=15, zorder=4, alpha=0.85)
ax_mds.set_aspect('equal')
ax_mds.set_axis_off()
edge_collection_mds = LineCollection([], colors='#7f8c8d', linewidths=0.5, alpha=0.2, zorder=1)
ax_mds.add_collection(edge_collection_mds)

# B) Panel Derecho: Mapa de la CDMX
scatter_geo = ax_geo.scatter(points_geo[:, 0], points_geo[:, 1], c=labels_iniciales, 
                             cmap='turbo', s=15, zorder=4, alpha=0.85)

valid_lons = points_geo[np.isfinite(points_geo[:, 0]), 0]
valid_lats = points_geo[np.isfinite(points_geo[:, 1]), 1]
margen = 0.02 
ax_geo.set_xlim(np.min(valid_lons) - margen, np.max(valid_lons) + margen)
ax_geo.set_ylim(np.min(valid_lats) - margen, np.max(valid_lats) + margen)
ax_geo.set_aspect('equal')
ax_geo.set_axis_off()

ctx.add_basemap(ax_geo, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron, alpha=0.7)
edge_collection_geo = LineCollection([], colors='#7f8c8d', linewidths=0.5, alpha=0.2, zorder=1)
ax_geo.add_collection(edge_collection_geo)

# C) Construcción del Slider Original Centrado abajo
ax_slider = plt.axes([0.25, 0.06, 0.50, 0.03], facecolor='#ecf0f1')
slider_tiempo = Slider(
    ax=ax_slider,
    label='Tiempo Traslado (min) ',
    valmin=0.0,
    valmax=limite_slider,
    valinit=0.0,
    valfmt='%1.2f min',
    color='#2980b9'
)

# --- 6. FUNCIÓN DE ACTUALIZACIÓN DEL SLIDER (CON CORRECCIÓN H0) ---
def update(val):
    epsilon = slider_tiempo.val
    
    # Tolerancia para vencer la pérdida de precisión de decimales flotantes en Python
    tol = 1e-5 
    
    # Segmentación de clusters (Distancia = 2 * epsilon + tolerancia)
    labels = sch.fcluster(Z, t=(2 * epsilon) + tol, criterion='distance')
    
    # Conteo dinámico de componentes activas
    n_islas_inf = np.sum(~np.isfinite(h0_data[:, 1]))
    n_clusters_actuales = max(1, len(np.unique(labels)) - n_islas_inf + 1)
    
    # Forzar estéticamente el cierre a 1 componente al llegar al límite
    if epsilon >= (limite_slider - 0.05):
        n_clusters_actuales = 1
        
    ax_mds.set_title("Espacio Métrico de Tiempos (MDS)\nLas distancias reflejan minutos de traslado reales", 
                     fontsize=12, fontweight='bold', pad=10)
    ax_geo.set_title(f"Traducción Geográfica (CDMX Map)\nComponentes Activas en la Red = {n_clusters_actuales}", 
                     fontsize=12, fontweight='bold', pad=10)
    
    for sc in [scatter_mds, scatter_geo]:
        sc.set_array(labels)
        sc.set_clim(vmin=labels.min(), vmax=labels.max())
    
    # Dibujar aristas aplicando la tolerancia
    ij_conexiones = np.argwhere((matrix_para_tda <= (2 * epsilon) + tol) & (matrix_para_tda > 0))
    if len(ij_conexiones) > 0:
        ij_conexiones = ij_conexiones[ij_conexiones[:, 0] < ij_conexiones[:, 1]]
        
        if len(ij_conexiones) > 8000:
            indices_muestra = np.random.choice(len(ij_conexiones), 8000, replace=False)
            ij_conexiones = ij_conexiones[indices_muestra]
            
        p_start_mds = points_mds[ij_conexiones[:, 0]]
        p_end_mds = points_mds[ij_conexiones[:, 1]]
        edge_collection_mds.set_segments(np.stack((p_start_mds, p_end_mds), axis=1))
        
        p_start_geo = points_geo[ij_conexiones[:, 0]]
        p_end_geo = points_geo[ij_conexiones[:, 1]]
        mask_valid = np.isfinite(p_start_geo).all(axis=1) & np.isfinite(p_end_geo).all(axis=1)
        edge_collection_geo.set_segments(np.stack((p_start_geo[mask_valid], p_end_geo[mask_valid]), axis=1))
    else:
        edge_collection_mds.set_segments([])
        edge_collection_geo.set_segments([])
        
    fig.suptitle(f"Filtración TDA de Infraestructura Hospitalaria CDMX (Radio $\epsilon$ = {epsilon:.2f} min)", 
                 fontsize=15, fontweight='bold', y=0.96)
    
    fig.canvas.draw_idle()

slider_tiempo.on_changed(update)
update(0.0)

print("-> Explorador interactivo listo para tu presentación.")
plt.show()