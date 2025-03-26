import requests

client_id = "dhrs76FHmGCYaFeSVgATpQ"
client_secret = "wrwzOuTyKCvi_fa9E1JvWet5qntqoQ"
username = "caothinh03"
password = "@condilon"
user_agent = "checkvote"

auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
data = {
    "grant_type": "password",
    "username": username,
    "password": password
}
headers = {"User-Agent": user_agent}

response = requests.post(
    "https://www.reddit.com/api/v1/access_token",
    auth=auth,
    data=data,
    headers=headers
)

token = response.json()["access_token"]
print( token)