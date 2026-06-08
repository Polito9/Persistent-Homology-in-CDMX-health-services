from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from ripser import ripser
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

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
    
    # Por una penalización finita muy alta:
    penalizacion_desconexion = 9999.0 
    asym_mat = df_asimetrica.fillna(penalizacion_desconexion).to_numpy(copy=True)
    
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

num_std = 2.0 
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
    print("df_tabla_h1.to_string(index=False, justify='center')")
    print(df_tabla_h1.info)
else:
    print("No se encontraron ciclos H1 significativos con el umbral actual.")
# --- ANÁLISIS DE MESETAS DE ESTABILIDAD TOPOLÓGICA (ESTADOS H0) ---

# 1. Obtener todos los tiempos exactos donde ocurre una fusión (muerte de H0)
tiempos_muerte = h0_data[np.isfinite(h0_data[:, 1]), 1]

# Añadimos el tiempo 0 (inicio absoluto) y ordenamos. Eliminamos duplicados por si 
# múltiples fusiones ocurren exactamente en el mismo milisegundo.
tiempos_eventos = np.unique(np.concatenate(([0.0], np.sort(tiempos_muerte))))

# 2. Calcular la duración de cada estado (Delta t)
# Cada elemento es el tiempo que el sistema pasó con un número constante de componentes
duraciones_estados = np.diff(tiempos_eventos)

# 3. Encontrar los Top 5 estados que duraron más tiempo sin cambiar
top_n_estados = 5
# argsort ordena de menor a mayor, [::-1] lo invierte, [:5] toma los primeros 5
indices_mesetas = np.argsort(duraciones_estados)[::-1][:top_n_estados]

print("\n" + "="*70)
print(" ANÁLISIS DE MESETAS: ESTADOS TOPOLÓGICOS MÁS ESTABLES (H0)")
print("="*70)

# Lista maestra que guardará los DataFrames de cada estado (tu "tabla de tablas")
lista_de_tablas_por_estado = []

for rank, idx in enumerate(indices_mesetas):
    t_inicio = tiempos_eventos[idx]
    t_fin = tiempos_eventos[idx + 1]
    duracion = duraciones_estados[idx]
    
    # Para saber qué nodos están conectados en este estado, tomamos un tiempo (epsilon) 
    # que esté justo en medio de la meseta temporal.
    epsilon_meseta = t_inicio + (duracion / 2.0)
    
    # Construimos la red de conexiones activa en ese momento específico
    matriz_estado_actual = (matrix_para_tda <= epsilon_meseta)
    
    # Calculamos los componentes conectados en ese instante
    n_componentes, etiquetas_nodos = connected_components(matriz_estado_actual, directed=False)
    
    print(f"\n[ TOP {rank + 1} ESTADO MÁS ESTABLE ]")
    print(f" • Intervalo de tiempo: {t_inicio:.2f} a {t_fin:.2f} minutos")
    print(f" • Persistencia del estado: {duracion:.2f} minutos continuos")
    print(f" • El mapa se mantuvo congelado en exactamente: {n_componentes} componentes distintos")
    
    # Construimos la tabla interna para este estado específico
    datos_estado = []
    for comp_id in range(n_componentes):
        # Encontrar qué índices numéricos pertenecen a este grupo
        indices_grupo = np.where(etiquetas_nodos == comp_id)[0]
        
        # Mapear los índices numéricos a los nombres/IDs reales de los nodos
        try:
            nombres_grupo = [nodos_totales[i] for i in indices_grupo]
        except NameError:
            nombres_grupo = indices_grupo.tolist()
            
        datos_estado.append({
            "ID_Componente": comp_id,
            "Tamaño": len(nombres_grupo),
            "Nodos_en_este_Componente": str(nombres_grupo)
        })
        
    df_estado = pd.DataFrame(datos_estado)
    # Ordenar la tabla para ver los clústeres más grandes primero
    df_estado = df_estado.sort_values(by="Tamaño", ascending=False).reset_index(drop=True)
    
    # Guardamos la tabla en nuestra lista maestra
    lista_de_tablas_por_estado.append({
        "info_estado": f"Top {rank+1}: {duracion:.2f} min ({n_componentes} comp)",
        "dataframe": df_estado
    })
    
    # Imprimimos la tabla en la consola. 
    # Limitamos visualmente a los primeros 10 grupos para no inundar la terminal si hay muchos nodos aislados
    print(df_estado.head(10).to_string(index=False))
    if n_componentes > 10:
        print(f"  ... y {n_componentes - 10} componentes aislados más (tamaño 1).")
    print("-" * 70)

# LÍMITES DE VISTA
if len(h1_data) > 0:
    tiempos_muerte_h1 = h1_data[np.isfinite(h1_data[:, 1]), 1]
    max_muerte_h1 = np.max(tiempos_muerte_h1) if len(tiempos_muerte_h1) > 0 else np.max(matrix_para_mds)
else:
    max_muerte_h1 = np.max(matrix_para_mds)

limite_vista = max_muerte_h1 * 1.05


# CONFIGURACIÓN DE LA GRÁFICA PARA ANIMACIÓN
fig, (ax_mds, ax_pers) = plt.subplots(1, 2, figsize=(16, 7))
plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.3)

ax_mds.scatter(points[:, 0], points[:, 1], color='black', s=5, zorder=4, alpha=0.8)
ax_mds.set_aspect('equal')
ax_mds.set_xlim(np.min(points[:, 0]) - 5, np.max(points[:, 0]) + 5)
ax_mds.set_ylim(np.min(points[:, 1]) - 5, np.max(points[:, 1]) + 5)

# Colección para el "andamiaje" general (gris puro)
edge_collection_base = LineCollection([], colors='#b0b0b0', linewidths=0.5, alpha=0.15, zorder=1)
# Colección para resaltar los ciclos H1 activos (cyan brillante)
edge_collection_h1 = LineCollection([], colors='cyan', linewidths=2.0, alpha=0.9, zorder=3)

ax_mds.add_collection(edge_collection_base)
ax_mds.add_collection(edge_collection_h1)

ax_pers.plot([0, limite_vista], [0, limite_vista], color='gray', linestyle='--', alpha=0.5)
ax_pers.set_xlabel("Tiempo de Nacimiento (minutos)")
ax_pers.set_ylabel("Tiempo de Muerte (minutos)")
ax_pers.set_xlim(-1, limite_vista)
ax_pers.set_ylim(-1, limite_vista)

scat_h0_ruido = ax_pers.scatter([], [], color='gray', s=10, alpha=0.3)
scat_h1_ruido = ax_pers.scatter([], [], color='gray', s=15, marker='^', alpha=0.3)
scat_h0_sig = ax_pers.scatter([], [], color='darkorange', s=50, label='H0 Significativo', alpha=0.9, edgecolor='black', zorder=4)
scat_h1_sig = ax_pers.scatter([], [], color='purple', s=70, label='H1 Significativo', marker='^', alpha=0.9, edgecolor='black', zorder=4)
ax_pers.legend(loc='lower right')


# ANIMACIÓN DINÁMICA
num_frames = 60
valores_epsilon = np.linspace(0.0, limite_vista, num_frames)

def animate(frame):
    epsilon = valores_epsilon[frame]
    
    # --- 1. DIBUJAR ANDAMIAJE (Grafo General) ---
    ax_mds.set_title(f"Complejo Simplicial (Mapa MDS)\nRadio = {epsilon:.2f} min", fontsize=11, fontweight='bold')
    
    ij_conexiones = np.argwhere((matrix_para_tda <= 2 * epsilon) & (matrix_para_tda > 0))
    if len(ij_conexiones) > 0:
        ij_conexiones = ij_conexiones[ij_conexiones[:, 0] < ij_conexiones[:, 1]]
        p_start = points[ij_conexiones[:, 0]]
        p_end = points[ij_conexiones[:, 1]]
        
        if len(p_start) > 5000:
            indices_muestra = np.random.choice(len(p_start), 5000, replace=False)
            p_start = p_start[indices_muestra]
            p_end = p_end[indices_muestra]

        segments_base = np.stack((p_start, p_end), axis=1)
        edge_collection_base.set_segments(segments_base)
    else:
        edge_collection_base.set_segments([])

    # --- 2. DIBUJAR HUECOS H1 RESALTADOS ---
    segmentos_h1_activos = []
    for i, h1_punto in enumerate(h1_sig):
        nacimiento = h1_punto[0]
        muerte = h1_punto[1]
        
        # Si el hueco H1 está "vivo" en este epsilon
        if nacimiento <= epsilon < muerte:
            cociclo = cocycles_sig[i]
            # Extraer las coordenadas de los nodos formadores de la pared del hueco
            p_start_h1 = points[cociclo[:, 0].astype(int)]
            p_end_h1 = points[cociclo[:, 1].astype(int)]
            
            # Anexarlos a la lista maestra de líneas a dibujar en este cuadro
            seg_h1 = np.stack((p_start_h1, p_end_h1), axis=1)
            segmentos_h1_activos.extend(seg_h1)
            
    if len(segmentos_h1_activos) > 0:
        edge_collection_h1.set_segments(segmentos_h1_activos)
    else:
        edge_collection_h1.set_segments([])


    # --- 3. ACTUALIZAR DIAGRAMA DE PERSISTENCIA ---
    vivos_h0_sig = h0_sig[(h0_sig[:, 0] <= epsilon) & (h0_sig[:, 1] > epsilon)]
    vivos_h1_sig = h1_sig[(h1_sig[:, 0] <= epsilon) & (h1_sig[:, 1] > epsilon)] if len(h1_sig) > 0 else []
    
    ax_pers.set_title(f"Topología Significativa (>{num_std} std)\nH0 Activos: {len(vivos_h0_sig)} | H1 Activos (Cyan): {len(vivos_h1_sig)}", 
                      fontsize=11, fontweight='bold', color='darkred')

    def actualizar_scatter(scatter_obj, datos_completos):
        validos = datos_completos[datos_completos[:, 0] <= epsilon]
        if len(validos) > 0:
            vis = validos.copy()
            vis[~np.isfinite(vis)] = limite_vista * 0.95 
            scatter_obj.set_offsets(vis)
        else:
            scatter_obj.set_offsets(np.empty((0, 2)))

    actualizar_scatter(scat_h0_ruido, h0_ruido)
    actualizar_scatter(scat_h1_ruido, h1_ruido)
    actualizar_scatter(scat_h0_sig, h0_sig)
    actualizar_scatter(scat_h1_sig, h1_sig)
        
    return edge_collection_base, edge_collection_h1, scat_h0_ruido, scat_h1_ruido, scat_h0_sig, scat_h1_sig


# EXPORTACIÓN 
ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=350, blit=True)
nombre_gif = 'filtracion_tda_significativa_con_generadores.gif'
ani.save(nombre_gif, writer='pillow', fps=3)