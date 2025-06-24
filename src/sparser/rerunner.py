# to run the simulator multiple times with different parameters
import requests
import json 

def send_simulator_data(set_sim_data, set_move_patterns, set_pap_data, params): 
    matrices = params['matrices']
    location = params['location']
    hours = params['hours']
    pmask = params['pmask']
    pvaccine = params['pvaccine']
    capacity = params['capacity']
    lockdown = params['lockdown']
    selfiso = params['selfiso']
    randseed = params['randseed']
    zone = params['zone']
    use_cache = params['useCache']

    DB_URL = 'http://127.0.0.1:1890/'; 

    try: 
        patterns_url = f"{DB_URL}patterns/{zone['id']}"
        response = requests.get(patterns_url)

        if response.status_code == 200:
            json_data = response.json()
            pap_data = json_data['data']['papdata']
            set_pap_data(pap_data)
        else: 
            raise requests.exceptions.RequestException(f"HTTP {response.status_code}")
    except Exception as e: 
        print(f"Error fetching patterns data: {e}")
    
    post_data = {
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

    make_post_request(post_data, set_sim_data, set_move_patterns, use_cache)
    