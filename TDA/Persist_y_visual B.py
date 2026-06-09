from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from ripser import ripser
import folium
from folium.plugins import TimestampedGeoJson
import datetime

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

# Cargando datos
df = pd.read_csv('denue_inegi_62_.csv', encoding="latin-1")
filter_data = df[df["cve_ent"].isin([9])]

#Usando solo los servicios de atencion primaria/inmediata
codigos = [
    621111, 621112, 621115, 621116, 
    621491, 621492, 622111, 622112
]
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
mapa_animado = folium.Map(location=[centro_lat, centro_lon], zoom_start=11, tiles='OpenStreetMap')

print("Agregando nodos permanentes al mapa base...")
fg_nodos = folium.FeatureGroup(name="Nodos Iniciales (Servicios de Salud)", show=True)

for lon, lat in points_geo:
    folium.CircleMarker(
        location=[lat, lon],      # Folium requiere estrictamente [Latitud, Longitud]
        radius=3,                 # Tamaño del punto
        color='#1a1a1a',          # Color del borde del punto
        weight=1,
        fill=True,
        fill_color="#ff4949",     # Fondo blanco para resaltar sobre el mapa oscuro
        fill_opacity=0.8
    ).add_to(fg_nodos)

fg_nodos.add_to(mapa_animado)

# 1. CREAR LÍNEA DE TIEMPO FICTICIA 
num_frames = 60
valores_epsilon = np.linspace(0.0, limite_vista, num_frames)

# Simulamos que cada frame es 1 minuto de reloj
fecha_base = datetime.datetime(2024, 1, 1, 0, 0)
tiempos_frames = [(fecha_base + datetime.timedelta(minutes=i)).strftime('%Y-%m-%dT%H:%M:%S') for i in range(num_frames)]

features = []

# 2. FILTRAR Y MUESTREAR EL ANDAMIAJE (Líneas grises)
umbral_maximo = 2 * limite_vista
i_indices, j_indices = np.where((matrix_para_tda > 0) & (matrix_para_tda <= umbral_maximo))

# Evitar duplicados (A->B es lo mismo que B->A)
mask = i_indices < j_indices
i_indices = i_indices[mask]
j_indices = j_indices[mask]

# Límite estricto para evitar crashear el archivo HTML
max_lineas = 5000 
if len(i_indices) > max_lineas:
    indices_aleatorios = np.random.choice(len(i_indices), max_lineas, replace=False)
    i_indices = i_indices[indices_aleatorios]
    j_indices = j_indices[indices_aleatorios]

print("Generando datos temporales para el andamiaje base...")
for idx_nodo1, idx_nodo2 in zip(i_indices, j_indices):
    distancia = matrix_para_tda[idx_nodo1, idx_nodo2]
    
    # Encontramos en qué frame "nace" la conexión
    frame_nac = np.searchsorted(2 * valores_epsilon, distancia)
    
    if frame_nac < num_frames:
        # La línea estará "viva" desde que nace hasta el final del slider
        tiempos_activos = tiempos_frames[frame_nac:]
        
        lon1, lat1 = points_geo[idx_nodo1]
        lon2, lat2 = points_geo[idx_nodo2]
        
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': [[lon1, lat1], [lon2, lat2]] # Folium requiere [Lon, Lat] en GeoJSON
            },
            'properties': {
                'times': tiempos_activos,
                'style': {'color': "#3a3a3a", 'weight': 1, 'opacity': 0.3}
            }
        })

# 3. AGREGAR LOS HUECOS H1 SIGNIFICATIVOS (Ciclos Cyan)
print("Generando datos temporales para los huecos H1...")
for i, h1_punto in enumerate(h1_sig):
    nacimiento = h1_punto[0]
    muerte = h1_punto[1] if np.isfinite(h1_punto[1]) else limite_vista
    
    frame_nac = np.searchsorted(valores_epsilon, nacimiento)
    frame_muerte = np.searchsorted(valores_epsilon, muerte)
    
    if frame_nac < num_frames:
        # El ciclo H1 SÓLO existe entre su nacimiento y su muerte
        tiempos_activos_h1 = tiempos_frames[frame_nac:frame_muerte]
        
        if not tiempos_activos_h1:
            continue
            
        cociclo = cocycles_sig[i]
        for edge in cociclo:
            u, v = int(edge[0]), int(edge[1])
            lon_u, lat_u = points_geo[u]
            lon_v, lat_v = points_geo[v]
            
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[lon_u, lat_u], [lon_v, lat_v]]
                },
                'properties': {
                    'times': tiempos_activos_h1,
                    'style': {'color': "#75eeee", 'weight': 3.5, 'opacity': 0.9}
                }
            })

# 4. CONSTRUIR EL REPRODUCTOR Y EXPORTAR
print("Ensamblando el reproductor interactivo...")
TimestampedGeoJson(
    {'type': 'FeatureCollection', 'features': features},
    period='PT1M', # Paso de 1 minuto ficticio por cada salto del slider
    add_last_point=False,
    auto_play=True, # Empieza a reproducirse al abrir
    loop=False,
    max_speed=1,
    loop_button=True,
    date_options='HH:mm', # Oculta la fecha falsa, solo muestra los minutos
    time_slider_drag_update=True
).add_to(mapa_animado)

nombre_archivo_html = "topologia_cdmx_slider.html"
mapa_animado.save(nombre_archivo_html)
print(f"¡Terminado! Abre el archivo '{nombre_archivo_html}' en tu navegador web.")