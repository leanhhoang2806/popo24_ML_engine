import requests

# Define the URL of the endpoint
url = "http://localhost:8000/query"

# Define the payload
payload = {
    "text": "No, you cannot evict a tenant for refusing to pay rent. Evicting a tenant for this reason is not allowed"
}

# Define the headers
headers = {
    "Content-Type": "application/json"
}

# Send the POST request
response = requests.post(url, json=payload, headers=headers)

# Print the response
response_json = response.json()
print(response['response'])
