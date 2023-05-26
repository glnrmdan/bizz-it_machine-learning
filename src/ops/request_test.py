import base64
import requests
import matplotlib.pyplot as plt
import json
import numpy as np
from PIL import Image
import io

# request data
im = '/home/irizqy/ml_ws/bangkit-ws/data/test/1_20230510_161105.jpg'
with open(im, "rb") as f:
    im_bytes = f.read()        
im_b64 = base64.b64encode(im_bytes).decode("utf8")
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

# server get

data =  {
    "data": im_b64
}

requests.post('http://127.0.0.1:3000/ld_predict', data=json.dumps(data), headers=headers)
