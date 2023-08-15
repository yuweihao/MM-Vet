"""
Install bardapi by
```bash
pip install bardapi
```

Run this script by
```bash
while true; do python inference.py; sleep 60; done
```
Currently we have to use loop in bash instead of python because the Bard-API seems have a bug.

Remember to change your image folder and meta data path in this script.
"""

import pandas as pd
import os
import json
import time
from bardapi import Bard


BARD_TOKEN = "YOUR_TOKEN_HERE" # https://github.com/dsdanielpark/Bard-API#authentication

model_name = "bard"
bard_error = "Temporarily unavailable due to traffic or an error in cookie values."

# change the path to your own path
results_path = f'../results/{model_name}.json' # path to save the results
image_folder = f"/path/to/mm-vet/images" 
meta_data = "/path/to/mm-vet/mm-vet.json" 


with open(meta_data, 'r') as f:
    data = json.load(f)


if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)
else:
    results = {}

data_num = len(data)

# time.sleep(60)
for i in range(len(data)):
    id = f"v1_{i}"
    if id in results and not (bard_error in results[id]):
        continue
    # time.sleep(60)
    imagename = data[id]['imagename']
    img_path = os.path.join(image_folder, imagename)
    prompt = data[id]['question']
    prompt = prompt.strip()
    print(f"\nPrompt: {prompt}")
    # load sample image
    bard = Bard(token=BARD_TOKEN)
    image = open(img_path, 'rb').read() # (jpeg, png, webp) are supported.
    bard_answer = bard.ask_about_image(prompt, image)
    response = bard_answer['content']
    if bard_error in response:
        time.sleep(60)
        break
    
    print(f"Response: {response}")
    results[id] = response        
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    break