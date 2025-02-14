import json
import numpy as np

def loadFromJSON(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = np.array(value)
    
    return data

