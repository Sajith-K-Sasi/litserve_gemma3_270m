import requests, json

data = {"prompt": "Write a poem in 10 lines about the beauty of nature"}

response = requests.post(url="http://127.0.0.1:8000/predict", json=data, stream=True)

for line in response.iter_lines(decode_unicode=True):
    if line:
        print(json.loads(line)["output"], end="")