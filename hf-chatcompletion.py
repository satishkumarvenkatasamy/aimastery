import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hyperbolic",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "what did Modi do in phalgam attack?"
        }
    ],
)

print(completion.choices[0].message)

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B:groq",
    messages=[
        {
            "role": "user",
            "content": "what did Modi do in phalgam attack?"
        }
    ],
)

print(completion.choices[0].message)

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    messages=[
        {
            "role": "user",
            "content": "what did Modi do in phalgam attack?"
        }
    ],
)

print(completion.choices[0].message)
