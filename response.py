import requests

files = {'file': open('example.jpg', 'rb')}
response = requests.post("https://YOUR_USERNAME-your-space-name.hf.space/run/predict", files=files)

print(response.json())
