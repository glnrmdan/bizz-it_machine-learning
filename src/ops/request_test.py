import base64
import requests
import json


# request data
im = '/home/irizqy/Downloads/ct_5.png'
# im = '/home/irizqy/ml_ws/bangkit-ws/data/bizit-dev_test-data/32.jpg'
# im = '/home/irizqy/Downloads/rfc.jpg'

with open(im, "rb") as f:
    im_bytes = f.read()        
im_b64 = base64.b64encode(im_bytes).decode("utf8")
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

# server get
data =  {
    "image": im_b64
}

req = requests.post('http://127.0.0.1:3000/ld_predict', data=json.dumps(data), headers=headers)
print(req.json())