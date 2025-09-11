import requests

url = "http://localhost:8000"

query = "can you recommend me a black shirt"

response = requests.post(url + "/recommend", json={"question": query})

print(response.json())