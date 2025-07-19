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

#
api_key_filename = "/Users/satishkumar/openai.key"
with open(api_key_filename, "r") as apikeyfile:
    openai_api_key = apikeyfile.read()
    openai_api_key = openai_api_key.replace("\n", "")

headers = {"Authorization": "Bearer "+openai_api_key,
           "Content-Type": "application/json"}

'''
curl https://api.openai.com/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4.1",
    "input": "Tell me a three sentence bedtime story about a unicorn."
      }'
'''

rest_api_endpoint = "https://api.openai.com/v1/chat/completions"

model_name = "gpt-4o"
your_ask = "Explain what is graph database in 5 bulleted points."

body = {
    "model": model_name,
    "messages": [
        {"role": "user", "content": your_ask}
    ]
}

client = httpx.Client(headers=headers)
response = client.post(rest_api_endpoint, json=body, timeout=10.0)

if response.status_code != 200:
    print("Request failed")
    print(response)
    exit(1)

response_json = response.json()


'''
Code to print all keys and values of response_json
Uncomment the following block
'''
'''
print("-"*50)
for key in response_json.keys():
    print(key, ":", response_json[key])
    print("-"*25)
print("-"*50)
'''

print("Answer is:")
print(response_json['choices'][0]['message']['content'])