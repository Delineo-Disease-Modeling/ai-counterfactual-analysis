import requests
import json


def make_post_request(req_data, set_sim_data, set_move_patterns, use_cache, db_url, sim_url):
    if use_cache:
        try:
            resp = requests.get(f"{db_url}simdata/{req_data['czone_id']}")
            
            if resp.status_code == 200 and resp.json().get('data'):
                print('Using cached data!')
                
                data = json.loads(resp.json()['data'])
                movement = data['movement']
                
                # Remove movement data beyond the specified length
                keys_to_remove = [key for key in movement.keys() if int(key) > req_data['length']]
                for key in keys_to_remove:
                    del movement[key]
                
                set_sim_data(data['result'])
                set_move_patterns(movement)
                return
                
        except requests.RequestException as error:
            print(f"Cache request error: {error}")
    
    try:
        response = requests.post(f"{sim_url}simulation/", json=req_data)
        
        if response.status_code != 200:
            raise Exception('Status code mismatch')
        
        data = response.json()
        if not data.get('result'):
            raise Exception('Invalid JSON (missing result)')
        
        set_sim_data(data['result'])
        set_move_patterns(data['movement'])
        
    except requests.RequestException as error:
        print(f"Simulation request error: {error}")
    except Exception as error:
        print(f"Error: {error}")


def send_simulator_data(set_sim_data, set_move_patterns, set_pap_data, 
                       matrices, location, hours, pmask, pvaccine, capacity, 
                       lockdown, selfiso, randseed, zone, use_cache, 
                       db_url, sim_url):
    # Fetch patterns data
    try:
        response = requests.get(f"{db_url}patterns/{zone.id}")
        
        if not response.ok:
            raise requests.RequestException("Failed to fetch patterns")
        
        json_data = response.json()
        set_pap_data(json_data['data']['papdata'])
        
    except requests.RequestException as error:
        print(f"Patterns request error: {error}")
    
    # Prepare request data for simulation
    req_data = {
        'czone_id': zone.id,
        'matrices': matrices,
        'location': location,
        'length': hours * 60,  # Convert hours to minutes
        'mask': pmask,
        'vaccine': pvaccine,
        'capacity': capacity,
        'lockdown': lockdown,
        'selfiso': selfiso,
        'randseed': randseed
    }
    
    # Make the simulation request
    make_post_request(req_data, set_sim_data, set_move_patterns, use_cache, db_url, sim_url)


# Example usage:
if __name__ == "__main__":
    # Mock callback functions for demonstration
    def mock_set_sim_data(data):
        print(f"Setting sim data: {len(data) if data else 'None'} items")
    
    def mock_set_move_patterns(data):
        print(f"Setting move patterns: {len(data) if data else 'None'} items")
    
    def mock_set_pap_data(data):
        print(f"Setting PAP data: {len(data) if data else 'None'} items")
    
    # Mock zone object
    class MockZone:
        def __init__(self, zone_id):
            self.id = zone_id
    
    # Example function call
    # send_simulator_data(
    #     set_sim_data=mock_set_sim_data,
    #     set_move_patterns=mock_set_move_patterns,
    #     set_pap_data=mock_set_pap_data,
    #     matrices=[],
    #     location={},
    #     hours=24,
    #     pmask=0.8,
    #     pvaccine=0.7,
    #     capacity=100,
    #     lockdown=False,
    #     selfiso=True,
    #     randseed=12345,
    #     zone=MockZone("zone_123"),
    #     use_cache=True,
    #     db_url="https://api.example.com/",
    #     sim_url="https://sim.example.com/"
    # )