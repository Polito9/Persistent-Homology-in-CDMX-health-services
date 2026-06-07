import pandas as pd 
import folium
import requests
import polyline
import urllib.parse

#Cargando informacion
data = pd.read_csv("denue_inegi_62_.csv", encoding="latin-1")

# --- DATA CLEANING

#Usando data de ciudad de mexico
filter_data = data[data["cve_ent"].isin([9])]

#Usando solo los servicios de atencion primaria/inmediata
codigos = [
    621111, 621112, 621115, 621116,
    621491, 621492, 622111, 622112
]
df = filter_data[filter_data['codigo_act'].isin(codigos)]


# -- LOCAL API CONSULT (Need the preconfiguration of orsm in docker)

lat_lon_points = df[['latitud', 'longitud']].values.tolist()
lista_ids = df['id'].tolist()

# Comprimir las coordenadas usando polyline
encoded_string = polyline.encode(lat_lon_points)

safe_polyline = urllib.parse.quote(encoded_string)

# --- Petición POST con Form-Data ---
url = f"http://localhost:5000/table/v1/driving/polyline({safe_polyline})"

# Los parámetros se envían como un diccionario de formulario
params = {
    "annotations": "duration,distance"
}

try:
    # CLAVE: Usar 'data=payload' hace que Requests envíe la petición como form-urlencoded
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    # --- 4. Extraer matrices y asignar tus IDs ---
    if "durations" in data and "distances" in data:
        print("Matriz calculada con éxito. Procesando DataFrames...")
        
        # Mapeamos los IDs a las filas y columnas automáticamente
        df_tiempos_matriz = pd.DataFrame(data["durations"], index=lista_ids, columns=lista_ids)
        df_distancias_matriz = pd.DataFrame(data["distances"], index=lista_ids, columns=lista_ids)
        
        # --- 5. Transformar a tabla estructurada (Formato Largo) ---
        df_tiempos_long = df_tiempos_matriz.reset_index().melt(
            id_vars='index', var_name='id_destino', value_name='tiempo_segundos'
        ).rename(columns={'index': 'id_origen'})
        
        df_distancias_long = df_distancias_matriz.reset_index().melt(
            id_vars='index', var_name='id_destino', value_name='distancia_metros'
        ).rename(columns={'index': 'id_origen'})
        
        # Unimos tiempos y distancias en una sola tabla
        df_final = pd.merge(df_tiempos_long, df_distancias_long, on=['id_origen', 'id_destino'])
        
        # --- 6. Guardar ---
        df_final.to_csv("matriz_origen_destino.csv", index=False)
        print(f"Archivo guardado exitosamente con {len(df_final)} filas.")
        
    else:
        print("Error: OSRM no devolvió las claves esperadas.")

except requests.exceptions.RequestException as e:
    print(f"Error de conexión: {e}")