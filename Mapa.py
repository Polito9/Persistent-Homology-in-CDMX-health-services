from pathlib import Path
import contextily as ctx
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


    nodos_totales = df_ponderada.index.tolist()
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
points_geo = evecs[:, idx[:2]] * np.sqrt(evals_top)


# TDA AVANZADO (Con extracción de Generadores / Cociclos)
# Activamos do_cocycles=True para obtener los nodos que forman los H1
resultado_ripser = ripser(matrix_para_tda, distance_matrix=True, maxdim=1, do_cocycles=True)
dgms = resultado_ripser['dgms']
cocycles_h1_totales = resultado_ripser['cocycles'][1] if len(resultado_ripser['cocycles']) > 1 else []

h0_data = dgms[0]
h1_data = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

def filtrar_por_desviacion_con_indices(datos, desviaciones=1.0):
    """Filtra y además devuelve los índices originales para rastrear los cociclos."""
    if len(datos) == 0:
        return datos, np.empty((0, 2)), 0.0, []
        
    finitos = datos[np.isfinite(datos[:, 1])]
    if len(finitos) == 0:
        return datos, np.empty((0, 2)), 0.0, []
        
    persistencia = finitos[:, 1] - finitos[:, 0]
    media = np.mean(persistencia)
    desviacion_std = np.std(persistencia)
    umbral = media + (desviaciones * desviacion_std)
    
    with np.errstate(invalid='ignore'):
        persistencia_total = datos[:, 1] - datos[:, 0]
    
    es_infinito = np.isinf(datos[:, 1])
    es_significativo = (persistencia_total >= umbral) | es_infinito
    
    significativos = datos[es_significativo]
    ruido = datos[~es_significativo]
    indices_significativos = np.where(es_significativo)[0]
    
    return significativos, ruido, umbral, indices_significativos

num_std = 1.0 
h0_sig, h0_ruido, umbral_h0, _ = filtrar_por_desviacion_con_indices(h0_data, num_std)
h1_sig, h1_ruido, umbral_h1, indices_h1_sig = filtrar_por_desviacion_con_indices(h1_data, num_std)

# --- GENERACIÓN DE TABLA DE NODOS H1 ---
tabla_h1 = []
cocycles_sig = [] # Guardaremos solo los cociclos de los H1 importantes para la animación

for idx_local, idx_global in enumerate(indices_h1_sig):
    nacimiento = h1_sig[idx_local, 0]
    muerte = h1_sig[idx_local, 1]
    
    # Extraemos el cociclo representativo del H1
    cociclo_actual = cocycles_h1_totales[idx_global]
    cocycles_sig.append(cociclo_actual)
    
    # Extraer los índices únicos de los nodos involucrados (columnas 0 y 1 del cociclo)
    nodos_indices = np.unique(cociclo_actual[:, :2].astype(int))
    
    # Convertimos los índices a los IDs reales si 'nodos_totales' existe, sino usamos el índice
    try:
        nombres_nodos = [nodos_totales[idx] for idx in nodos_indices]
    except NameError:
        nombres_nodos = nodos_indices.tolist()
        
    tabla_h1.append({
        "H1_ID": f"Hueco {idx_local + 1}",
        "Nacimiento": round(nacimiento, 2),
        "Muerte": round(muerte, 2),
        "Persistencia": round(muerte - nacimiento, 2),
        "Nodos_Formadores": str(nombres_nodos)
    })

df_tabla_h1 = pd.DataFrame(tabla_h1)
print(f"\n--- ANÁLISIS DE PERSISTENCIA (>{num_std} std) ---")
print(f"H0 (Componentes) Significativos: {len(h0_sig)}")
print(f"H1 (Huecos/Ciclos) Significativos: {len(h1_sig)}")
print("\n--- NODOS ASOCIADOS A HUECOS H1 ---")
# Imprimimos la tabla formateada para que sea legible en consola
if not df_tabla_h1.empty:
    print(df_tabla_h1.to_string(index=False, justify='center'))
else:
    print("No se encontraron ciclos H1 significativos con el umbral actual.")


# LÍMITES DE VISTA
if len(h1_data) > 0:
    tiempos_muerte_h1 = h1_data[np.isfinite(h1_data[:, 1]), 1]
    max_muerte_h1 = np.max(tiempos_muerte_h1) if len(tiempos_muerte_h1) > 0 else np.max(matrix_para_mds)
else:
    max_muerte_h1 = np.max(matrix_para_mds)

limite_vista = max_muerte_h1 * 1.05



#Obtener el archivo con lat y long
df = pd.read_csv('denue_inegi_62_.csv', encoding="latin-1")
filter_data = df[df["cve_ent"].isin([9])]

#Usando solo los servicios de atencion primaria/inmediata
codigos = [
    621111, 621112, 621115, 621116, 
    621491, 621492, 622111, 622112
]
df_coords = filter_data[filter_data['codigo_act'].isin(codigos)]

df_coords = df_coords.set_index('id')
#print("Df coords:")
#print(df_coords)
#print("Nodos totales:")
#print(nodos_totales)
df_coords = df_coords.reindex(nodos_totales)
#print("Df coords 2: ")
#print(df_coords)
lats = df_coords['latitud'].values
lons = df_coords['longitud'].values

# Reemplazamos la variable 'points' del MDS por los puntos geográficos
# X = Longitud, Y = Latitud
points_geo = np.column_stack((lons, lats))
points_geo = points_geo[np.isfinite(points_geo).all(axis=1)]

# --- CONFIGURACIÓN DE LA GRÁFICA PARA ANIMACIÓN (SOLO MAPA) ---
fig, ax_geo = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)

# Dibujamos los puntos geográficos (Nodos)
ax_geo.scatter(points_geo[:, 0], points_geo[:, 1], color='black', s=8, zorder=4, alpha=0.8)

# Configuramos límites basados en las coordenadas reales con un margen adecuado (0.02 grados ~ 2 km)
margen = 0.02 
ax_geo.set_xlim(np.min(points_geo[:, 0]) - margen, np.max(points_geo[:, 0]) + margen)
ax_geo.set_ylim(np.min(points_geo[:, 1]) - margen, np.max(points_geo[:, 1]) + margen)

# Mantener la proporción real del mapa y ocultar ejes
ax_geo.set_aspect('equal')
ax_geo.set_axis_off()

# Añadimos el mapa base de la CDMX
ctx.add_basemap(ax_geo, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron, alpha=0.7)

# Colección para el "andamiaje" general (gris puro)
edge_collection_base = LineCollection([], colors='#b0b0b0', linewidths=0.8, alpha=0.3, zorder=1)
# Colección para resaltar los ciclos H1 activos (cyan brillante)
edge_collection_h1 = LineCollection([], colors='cyan', linewidths=2.5, alpha=1.0, zorder=3)

ax_geo.add_collection(edge_collection_base)
ax_geo.add_collection(edge_collection_h1)

# --- ANIMACIÓN DINÁMICA ---
num_frames = 60
valores_epsilon = np.linspace(0.0, limite_vista, num_frames)

def animate(frame):
    epsilon = valores_epsilon[frame]
    
    # 1. Título dinámico
    ax_geo.set_title(f"Evolución Topológica en CDMX\nTiempo máximo de traslado = {epsilon:.2f} min", 
                     fontsize=14, fontweight='bold')
    
    # 2. DIBUJAR ANDAMIAJE (Grafo General)
    ij_conexiones = np.argwhere((matrix_para_tda <= 2 * epsilon) & (matrix_para_tda > 0))
    if len(ij_conexiones) > 0:
        ij_conexiones = ij_conexiones[ij_conexiones[:, 0] < ij_conexiones[:, 1]]
        p_start = points_geo[ij_conexiones[:, 0]]
        p_end = points_geo[ij_conexiones[:, 1]]
        
        # Muestreo si hay demasiadas líneas para que no explote la RAM/GIF
        if len(p_start) > 6000:
            indices_muestra = np.random.choice(len(p_start), 6000, replace=False)
            p_start = p_start[indices_muestra]
            p_end = p_end[indices_muestra]

        segments_base = np.stack((p_start, p_end), axis=1)
        edge_collection_base.set_segments(segments_base)
    else:
        edge_collection_base.set_segments([])

    # 3. DIBUJAR HUECOS H1 RESALTADOS
    segmentos_h1_activos = []
    for i, h1_punto in enumerate(h1_sig):
        nacimiento = h1_punto[0]
        muerte = h1_punto[1]
        
        # Si el hueco H1 está "vivo" en este epsilon
        if nacimiento <= epsilon < muerte:
            cociclo = cocycles_sig[i]
            p_start_h1 = points_geo[cociclo[:, 0].astype(int)]
            p_end_h1 = points_geo[cociclo[:, 1].astype(int)]
            
            seg_h1 = np.stack((p_start_h1, p_end_h1), axis=1)
            segmentos_h1_activos.extend(seg_h1)
            
    if len(segmentos_h1_activos) > 0:
        edge_collection_h1.set_segments(segmentos_h1_activos)
    else:
        edge_collection_h1.set_segments([])
        
    return edge_collection_base, edge_collection_h1

print("Generando animación...")

# EXPORTACIÓN 
ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=300, blit=True)
nombre_gif = 'filtracion_tda_mapa_cdmx.gif'
ani.save(nombre_gif, writer='pillow', fps=4)

print(f"¡Listo! Animación guardada como {nombre_gif}")