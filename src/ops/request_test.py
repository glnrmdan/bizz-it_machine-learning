import base64
import requests
import json


# request data
im = '/home/irizqy/ml_ws/bangkit-ws/data/test/2_20230507_092258.jpg'
with open(im, "rb") as f:
    im_bytes = f.read()        
im_b64 = base64.b64encode(im_bytes).decode("utf8")
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
print(type(im_b64))

# server get
data =  {
    "data": im_b64
}

# requests.post('http://127.0.0.1:3000/ld_predict', data=json.dumps(data), headers=headers)
