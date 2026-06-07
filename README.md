# Persistent-Homology-in-CDMX-health-services

### How to use the API

Given that the topological analysis of urban or spatial networks requires knowing the actual "distance" (whether physical in meters or temporal in seconds) between thousands of nodes, the use of direct Euclidean distances is insufficient. 

To solve this and establish the foundations of the topological graphs, a local routing engine was implemented using **Open Source Routing Machine (OSRM)**. This appendix documents the step-by-step process to ensure the **total reproducibility** of the obtained data.

#### Infrastructure and Prerequisites

To calculate massive asymmetric matrices (e.g., 4000x4000 nodes, generating 16 million vehicular edges) without relying on third-party APIs or suffering blocks due to request limits, a containerized local environment was configured using .

* **Engine:** Docker Desktop.
* **Base language:** Python 3.8+ (Libraries: `requests`, `pandas`, `polyline`).
* **Critical Memory Configuration (WSL2 on Windows):**
  Processing the country-level graph requires managing massive memory loads. A hard limit was implemented in the `.wslconfig` file of the *host* system to prevent memory overflows (OOM kills):
  ```ini
  [wsl2]
  memory=6GB
  swap=12GB
  ```

#### Construction of the Metric Space (OSRM Graph)

Obtaining the edge weights (times and distances) required processing the raw street network into a Contraction Hierarchies format.

The OpenStreetMap cartographic dump in `.osm.pbf` format for the study region (Mexico) was downloaded through the public [Geofabrik](https://download.geofabrik.de) repository, which was saved at the root of the disk in a folder named osrm_data.

The road network was extracted using the car profile, translating the raw nodes into a directed graph:
```bash
docker run -t -v "C:\osrm_data:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/region-estudio.osm.pbf
```

To allow matrix calculation in logarithmic time, the route hierarchies were calculated:
```bash
docker run -t -v "C:\osrm_data:/data" osrm/osrm-backend osrm-contract /data/region-estudio.osrm
```

The local OSRM server was deployed by modifying the native parameter restriction (`--max-table-size 8000`) to allow massive matrix queries that would serve as input for the TDA:
```bash
docker run -t -i -p 5000:5000 -v "C:\osrm_data:/data" osrm/osrm-backend osrm-routed --max-table-size 8000 /data/region-estudio.osrm
```

Once the server is active, you can use the OSRM API queries normally by changing the public server to the local one, with the only limitation being the local processing speed of your computer, meaning it can be done without an internet connection. An example of this can be found in the `testAPI.py` file.