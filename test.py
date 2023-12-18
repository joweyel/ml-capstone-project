import os
import random
import requests

url = "http://localhost:9696/predict"


# choose example image randomly 
image_path = movie = random.choice([img for img in os.listdir("test_imgs") if img.endswith("jpg")])

with open(f"test_imgs/{image_path}", 'rb') as file:
    # Create a dictionary with the imge data
    files = {'file': (image_path, file, 'image/jpeg')}

    # Send the request with the image file
    response = requests.post(url, files=files)

response = response.json()
response["prob"] = 1 - response["prob"] if response["pred_class_name"] == "Bike" else response["prob"]

print(response)