# Persistent-Homology-in-CDMX-health-services

### Como usar la API

Dado que el análisis topológico de redes urbanas o espaciales requiere conocer la "distancia" real (ya sea física en metros o temporal en segundos) entre miles de nodos, el uso de distancias euclidianas directas resulta insuficiente. 

Para resolver esto y sentar las bases de los grafos topológicos, se implementó un motor de enrutamiento local utilizando **Open Source Routing Machine (OSRM)**. Este anexo documenta el proceso paso a paso para asegurar la **reproducibilidad total** de los datos obtenidos.

#### Infraestructura y Requisitos Previos

Para calcular matrices asimétricas masivas (ej. 4000x4000 nodos, generando 16 millones de aristas vehiculares) sin depender de APIs de terceros ni sufrir bloqueos por límite de peticiones, se configuró un entorno local contenerizado usando .

* **Motor:** Docker Desktop.
* **Lenguaje base:** Python 3.8+ (Librerías: `requests`, `pandas`, `polyline`).
* **Configuración Crítica de Memoria (WSL2 en Windows):**
  El procesamiento del grafo a nivel país requiere gestionar cargas de memoria masivas. Se implementó un límite duro en el archivo `.wslconfig` del sistema *host* para evitar desbordamientos de memoria (OOM kills):
  ```ini
  [wsl2]
  memory=6GB
  swap=12GB
  ```

#### Construcción del Espacio Métrico (Grafo de OSRM)

La obtención de las ponderaciones de las aristas (tiempos y distancias) requirió procesar la red de calles cruda hacia un formato de jerarquías de contracción (Contraction Hierarchies).

Se descargó el volcado cartográfico de OpenStreetMap en formato `.osm.pbf` de la región de estudio (México) mediante el repositorio público de [Geofabrik](https://download.geofabrik.de), el cuál se guardó en la raiz del disco en una carpeta llamada osrm_data.

Se extrajo la red vial utilizando el perfil de automóvil, traduciendo los nodos crudos a un grafo dirigido:
```bash
docker run -t -v "C:\osrm_data:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/region-estudio.osm.pbf
```

Para permitir el cálculo matricial en tiempos de orden logarítmico, se calcularon las jerarquías de rutas:
```bash
docker run -t -v "C:\osrm_data:/data" osrm/osrm-backend osrm-contract /data/region-estudio.osrm
```

Se desplegó el servidor local de OSRM modificando la restricción nativa de parámetros (`--max-table-size 8000`) para permitir consultas matriciales masivas que servirían de insumo para el TDA:
```bash
docker run -t -i -p 5000:5000 -v "C:\osrm_data:/data" osrm/osrm-backend osrm-routed --max-table-size 8000 /data/region-estudio.osrm
```

Ya que se tiene activo el servidor se pueden usar consultas se puede usar la API de OSRM con onormalidad cambiando el servidor publico por el local y con el único limitante tu velocidad de procesamiento local de tu computadora, por lo que se puede realizar sin necesidad de conexión a internet.  Un ejemplo de ello se puede encontrar en el archivo `testAPI.py`