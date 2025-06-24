import requests
import json
import os

SIM_URL = os.environ.get('VITE_SIM_URL', 'http://127.0.0.1:1870/')
ALG_URL = os.environ.get('VITE_ALG_URL', 'http://127.0.0.1:1880/')
DB_URL  = os.environ.get('VITE_DB_URL',  'http://127.0.0.1:1890/')


def make_post_request(reqdata, use_cache=True):
    """
    Sends POST request to simulation server or fetches cached data if available.
    Returns (sim_data, move_patterns)
    """
    if use_cache:
        try:
            cache_resp = requests.get(f"{DB_URL}simdata/{reqdata['czone_id']}")
            if cache_resp.status_code == 200 and 'data' in cache_resp.json():
                print("Using cached data!")
                cached = json.loads(cache_resp.json()['data'])
                movement = cached.get('movement', {})

                # Truncate movement data beyond simulation length
                movement = {
                    k: v for k, v in movement.items()
                    if int(k) <= reqdata['length']
                }

                return cached['result'], movement
        except Exception as e:
            print(f"Cache error: {e}")

    # If no cache or error, do POST request
    try:
        response = requests.post(f"{SIM_URL}simulation/", json=reqdata)
        response.raise_for_status()
        data = response.json()

        if 'result' not in data:
            raise ValueError("Invalid JSON (missing 'result')")

        return data['result'], data['movement']
    except Exception as e:
        print(f"Simulation request error: {e}")
        return None, None


def send_simulator_data(
    matrices, location, hours, pmask, pvaccine, capacity, lockdown,
    selfiso, randseed, zone, use_cache=True
):
    """
    Sends full simulation data and pattern request.
    Returns (sim_data, move_patterns, pap_data)
    """
    # Fetch PAP data
    pap_data = None
    try:
        resp = requests.get(f"{DB_URL}patterns/{zone['id']}")
        resp.raise_for_status()
        pap_data = resp.json().get('data', {}).get('papdata', None)
    except Exception as e:
        print(f"PAP data fetch error: {e}")

    # Prepare request data
    reqdata = {
        'czone_id': zone['id'],
        'matrices': matrices,
        'location': location,
        'length': hours * 60,
        'mask': pmask,
        'vaccine': pvaccine,
        'capacity': capacity,
        'lockdown': lockdown,
        'selfiso': selfiso,
        'randseed': randseed
    }

    sim_data, move_patterns = make_post_request(reqdata, use_cache)
    return sim_data, move_patterns, pap_data

if __name__ == "__main__":
    zone = {
    "id": "cz123",
    "name": "Downtown Zone",
    "created_at": "2024-06-15T12:00:00"
    }

    sim_data, move_patterns, pap_data = send_simulator_data(
        matrices={"contacts": [[1, 2], [3, 4]]},
        location="City Center",
        hours=24,
        pmask=0.7,
        pvaccine=0.8,
        capacity=0.5,
        lockdown=True,
        selfiso=True,
        randseed=42,
        zone=zone,
        use_cache=True
    )
  
