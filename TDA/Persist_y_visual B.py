from pathlib import Path
import numpy as np
import pandas as pd
from ripser import ripser
import folium
from folium.plugins import TimestampedGeoJson
import datetime

# --- NUEVAS IMPORTACIONES PARA DIJKSTRA (PARA EL PERÍMETRO) ---
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

archivo_crudo = Path('matriz_origen_destino.csv') 
archivo_ponderado = Path('matriz_tiempos_ponderada_simetrica.csv')

if archivo_ponderado.is_file():
    df_ponderada = pd.read_csv(archivo_ponderado, index_col=0)
    df_ponderada.columns = df_ponderada.columns.astype(type(df_ponderada.index[0]))
    matrix_para_tda = df_ponderada.values
    n = df_ponderada.shape[0]
    nodos_totales = df_ponderada.index.tolist()
else:
    # (Este bloque se mantiene igual, no necesita cambios)
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

# TDA AVANZADO (Con extracción de Generadores / Cociclos)
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
h1_sig, h1_ruido, umbral_h1, indices_h1_sig = filtrar_por_desviacion_con_indices(h1_data, num_std)

# --- GENERACIÓN DE TABLA DE NODOS H1 ---
tabla_h1 = []
cocycles_sig = [] 

for idx_local, idx_global in enumerate(indices_h1_sig):
    nacimiento = h1_sig[idx_local, 0]
    muerte = h1_sig[idx_local, 1]
    cociclo_actual = cocycles_h1_totales[idx_global]
    cocycles_sig.append(cociclo_actual)
    nodos_indices = np.unique(cociclo_actual[:, :2].astype(int))
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
# Imprimimos la tabla formateada
if not df_tabla_h1.empty:
    print(df_tabla_h1.to_string(index=False, justify='center'))
else:
    print("No se encontraron ciclos H1 significativos.")

# LÍMITES DE VISTA
if len(h1_data) > 0:
    tiempos_muerte_h1 = h1_data[np.isfinite(h1_data[:, 1]), 1]
    # Límite vista un poco mayor para ver el andamiaje
    max_muerte_h1 = np.max(tiempos_muerte_h1) if len(tiempos_muerte_h1) > 0 else np.max(matrix_para_tda)
else:
    max_muerte_h1 = np.max(matrix_para_tda)

# Ajuste el límite vista para ver más andamiaje como en su imagen
limite_vista = max_muerte_h1 * 2 

# Cargando datos DENUE
df = pd.read_csv('denue_inegi_62_.csv', encoding="latin-1")
filter_data = df[df["cve_ent"].isin([9])]
codigos = [621111, 621112, 621115, 621116, 621491, 621492, 622111, 622112]
df_coords = filter_data[filter_data['codigo_act'].isin(codigos)]
df_coords = df_coords.set_index('id')
df_coords = df_coords.reindex(nodos_totales)
lats = df_coords['latitud'].values
lons = df_coords['longitud'].values
points_geo = np.column_stack((lons, lats))
points_geo = points_geo[np.isfinite(points_geo).all(axis=1)]

# Graficando
centro_lat = np.mean(points_geo[:, 1])
centro_lon = np.mean(points_geo[:, 0])
# Un zoom_start menor para ver toda la CDMX como en su imagen
mapa_animado = folium.Map(location=[centro_lat, centro_lon], zoom_start=10, tiles='OpenStreetMap')

print("Agregando nodos permanentes al mapa base...")
fg_nodos = folium.FeatureGroup(name="Nodos Iniciales", show=True)
for lon, lat in points_geo:
    folium.CircleMarker(
        location=[lat, lon],
        radius=1.5, # Puntos más pequeños para CDMX densa
        color='#1a1a1a',
        weight=0.5,
        fill=True,
        fill_color="#ff4949",
        fill_opacity=0.7
    ).add_to(fg_nodos)
fg_nodos.add_to(mapa_animado)

# 1. CREAR LÍNEA DE TIEMPO FICTICIA 
num_frames = 60
valores_epsilon = np.linspace(0.0, limite_vista, num_frames)
fecha_base = datetime.datetime(2024, 1, 1, 0, 0)
tiempos_frames = [(fecha_base + datetime.timedelta(minutes=i)).strftime('%Y-%m-%dT%H:%M:%S') for i in range(num_frames)]
features = []

# 2. FILTRAR Y MUESTREAR EL ANDAMIAJE (Líneas grises)
print("Generando datos temporales para el andamiaje base...")
umbral_maximo = limite_vista
i_indices, j_indices = np.where((matrix_para_tda > 0) & (matrix_para_tda <= umbral_maximo))
mask = i_indices < j_indices
i_indices = i_indices[mask]
j_indices = j_indices[mask]
# Un límite más alto si es necesario, pero 5k es seguro para el navegador
max_lineas = 5000 
if len(i_indices) > max_lineas:
    indices_aleatorios = np.random.choice(len(i_indices), max_lineas, replace=False)
    i_indices = i_indices[indices_aleatorios]
    j_indices = j_indices[indices_aleatorios]
for idx_nodo1, idx_nodo2 in zip(i_indices, j_indices):
    distancia = matrix_para_tda[idx_nodo1, idx_nodo2]
    frame_nac = np.searchsorted(2 * valores_epsilon, distancia) # Escala 2*epsilon para andamiaje
    if frame_nac < num_frames:
        tiempos_activos = tiempos_frames[frame_nac:]
        lon1, lat1 = points_geo[idx_nodo1]
        lon2, lat2 = points_geo[idx_nodo2]
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': [[lon1, lat1], [lon2, lat2]]
            },
            'properties': {
                'times': tiempos_activos,
                'style': {'color': "#3a3a3a", 'weight': 1, 'opacity': 0.25} # Más transparencia
            }
        })

# --- FUNCIÓN DE DIJKSTRA (Copiar si no la tienes) ---
def obtener_ciclo_representativo(matrix, cociclo, nacimiento):
    """
    Transforma un 1-cociclo (telaraña) en un 1-ciclo (polígono Dijkstra)
    en el instante de su nacimiento.
    """
    u_cociclo = cociclo[:, 0].astype(int)
    v_cociclo = cociclo[:, 1].astype(int)
    # Encontrar la arista creadora
    distancias = matrix[u_cociclo, v_cociclo]
    idx_nac = np.argmin(distancias)
    u, v = u_cociclo[idx_nac], v_cociclo[idx_nac]
    
    # Crear un subgrafo con aristas existentes hasta el nacimiento
    # Sumamos un pequeño epsilon para errores de punto flotante
    mask = (matrix <= nacimiento + 1e-5)
    grafo_adyacencia = np.zeros_like(matrix)
    grafo_adyacencia[mask] = matrix[mask]
    
    # Eliminar la arista creadora para obligar a dar la vuelta entera
    grafo_adyacencia[u, v] = 0
    grafo_adyacencia[v, u] = 0
    
    # Calcular el camino más corto bordeando el hueco
    grafo_sparse = csr_matrix(grafo_adyacencia)
    dist_matrix, predecessors = shortest_path(
        csgraph=grafo_sparse, directed=False, indices=u, return_predecessors=True
    )
    # Reconstruir la lista secuencial de aristas del polígono
    camino = []
    nodo_actual = v
    while nodo_actual != u:
        if nodo_actual == -9999: # Prevención por si no existe ruta
            break
        camino.append(nodo_actual)
        nodo_actual = predecessors[nodo_actual]
    
    aristas_ciclo = []
    if len(camino) > 0 and nodo_actual == u:
        camino.append(u)
        camino.reverse() 
        camino.append(u) # Cerrar el ciclo
        # Formar los pares de aristas conectadas
        for k in range(len(camino) - 1):
            aristas_ciclo.append((camino[k], camino[k+1]))
    else:
        # Respaldo: si por la topología no se encuentra el perímetro exacto
        aristas_ciclo = [(u, v)]
        
    return aristas_ciclo

# --- 3. AGREGAR LOS HUECOS H1 SIGNIFICATIVOS (Polígonos Rojos Rellenos) ---
print("Generando datos temporales para los huecos H1 (Polígonos rellenados)...")
for i, h1_punto in enumerate(h1_sig):
    nacimiento = h1_punto[0]
    # Muerte escalada para que dure más, como en su andamiaje
    muerte = h1_punto[1] if np.isfinite(h1_punto[1]) else limite_vista
    
    frame_nac = np.searchsorted(valores_epsilon, nacimiento)
    frame_muerte = np.searchsorted(valores_epsilon, muerte)
    
    if frame_nac < num_frames:
        tiempos_activos_h1 = tiempos_frames[frame_nac:frame_muerte]
        
        if not tiempos_activos_h1:
            continue
            
        cociclo = cocycles_sig[i]
        
        # Extraemos las aristas del perímetro usando Dijkstra
        aristas_poligono = obtener_ciclo_representativo(matrix_para_tda, cociclo, nacimiento)
        
        # --- RECONSTRUIR EL POLÍGONO COMPLETO ORDENADO (Para relleno) ---
        lista_nodos_ordenada = []
        if len(aristas_poligono) > 1:
            # El Dijkstra nos da pares ordenados (u,v), (v,w), (w,x)... 
            # Tomamos el primer nodo de cada par para la secuencia
            lista_nodos_ordenada = [edge[0] for edge in aristas_poligono]
            # No necesitamos añadir el último nodo 'u' porque es igual al primero de la arista final
        else:
            # Si solo hay una arista, no podemos formar un polígono, así que no lo rellenamos
            continue

        coordenadas_poligono = []
        for nodo_idx in lista_nodos_ordenada:
            coordenadas_poligono.append([points_geo[nodo_idx][0], points_geo[nodo_idx][1]]) # [Lon, Lat]

        # ¡IMPORTANTE: GeoJSON requiere cerrar el polígono (primer punto == último punto)!
        coordenadas_poligono.append([points_geo[lista_nodos_ordenada[0]][0], points_geo[lista_nodos_ordenada[0]][1]])

        # Crear una única Feature de tipo POLYGON para todo el hueco
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon', # <--- CAMBIADO: De LineString a Polygon
                'coordinates': [coordenadas_poligono] # <--- Lista de anillos, el anillo exterior es [0]
            },
            'properties': {
                'times': tiempos_activos_h1,
                # --- ESTILO ACTUALIZADO CON RELLENO ---
                'style': {
                    'color': "#ff0000",       # Color del borde: Rojo brillante
                    'weight': 3, 
                    'opacity': 0.8,
                    'dashArray': '6, 6',       # Línea punteada
                    'fillColor': "#ffb3b3",   # <--- NUEVO: Color de relleno Rojo claro
                    'fillOpacity': 0.3         # <--- NUEVO: Opacidad de relleno baja (0.3)
                }
            }
        })

# 4. CONSTRUIR EL REPRODUCTOR Y EXPORTAR
print("Ensamblando el reproductor interactivo...")
TimestampedGeoJson(
    {'type': 'FeatureCollection', 'features': features},
    period='PT1M',
    add_last_point=False,
    auto_play=True,
    loop=False,
    max_speed=1,
    loop_button=True,
    date_options='HH:mm',
    time_slider_drag_update=True
).add_to(mapa_animado)

nombre_archivo_html = "topologia_cdmx_slider.html"
mapa_animado.save(nombre_archivo_html)
print(f"¡Terminado! Abre el archivo '{nombre_archivo_html}' en tu navegador web.")