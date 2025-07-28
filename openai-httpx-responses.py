import httpx
import os
from pathlib import Path

'''
Steps to execute this tool. Skip those steps that you had already done.

1. Install python from https://www.python.org/downloads/

2. Install pip using command:
python -m ensurepip --upgrade

3. Install required modules using any one of these commands:
   python -m pip install requests httpx aiohttp asyncio
   or
   pip install requests httpx aiohttp asyncio

4. Setup OpenAI API key by visiting https://platform.openai.com/chat [Sign-up using your personal gmail id]

5. Generate OpenAI API key and store it in ~/openai.key file.

6. Execute this code
python openai-httpx.py
'''

def log_request(request):
    print(f"Request event hook: {request.method} {request.url} - Waiting for response")

def log_response(response):
    request = response.request
    print(f"Response event hook: {request.method} {request.url} - Status {response.status_code}")


#
api_key_filename = "/Users/satishkumar/openai.key"
with open(api_key_filename, "r") as apikeyfile:
    openai_api_key = apikeyfile.read()
    openai_api_key = openai_api_key.replace("\n", "")

headers = {"Authorization": "Bearer "+openai_api_key,
           "Content-Type": "application/json"}

rest_api_endpoint = "https://api.openai.com/v1/responses"

model_name = "gpt-4o"
your_ask = "Who is the vice president of India?"

body = f"""{{ 
               "model": "{model_name}",
               "input": "{your_ask}"
            }}"""


client = httpx.Client(event_hooks={'request': [log_request], 'response': [log_response]}, headers=headers)
#client = httpx.Client(headers=headers)
response = client.post(rest_api_endpoint, data=body, timeout=10.0)

if response.status_code != 200:
    print("Request failed")
    print(response)
    exit(1)

response_json = response.json()


'''
Code to print all keys and values
'''
'''
print("-"*50)
for key in response_json.keys():
    print(key, ":", response_json[key])
    print("-"*25)
print("-"*50)
'''

print("Answer is:")
print(response_json['output'][0]['content'][0]['text'])
