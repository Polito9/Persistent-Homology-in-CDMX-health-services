import requests

points = [(-99.16151944, 19.29129336), (-99.13772284, 19.27673891), (-99.11550311, 19.25168289)]


coordinates = ""

for i, p in enumerate(points):
    coordinates += str(p[0])+","+str(p[1])
    if(i != len(points)-1):
        coordinates+=";"

print(coordinates)

url = f"http://router.project-osrm.org/table/v1/driving/{coordinates}?annotations=distance,duration"


try:
    response = requests.get(url)
    response.raise_for_status()

    data = response.json()


    if(data.get("code") == 'Ok'):
        print(data)

        durations = data.get("durations", [])
        distances = data.get("distances", [])

        for i in range(len(points)):
            for j in range(len(points)):
                if i != j:
                    time_sec = durations[i][j]
                    dist_met = distances[i][j]
                    print(f"Time from Point {i} to Point {j}: {time_sec / 60:.2f} minutes")
                    print(f"Distance from Point {i} to Point {j}: {dist_met / 1000:.2f} km")

        #seg = data["routes"][0]["duration"]
        #meters = data["routes"][0]["distance"]

        #duracion_minutos = seg / 60
        #distancia_km = meters / 1000
        
        #print(f"Travel time: {duracion_minutos:.2f} minutes")
        #print(f"Total distance: {distancia_km:.2f} km")
    else:
        print(f"API ERROR: {data.get('message', 'Desconocido')}")

except requests.exceptions.RequestException as e:
    print(f"Connection error: {e}")

