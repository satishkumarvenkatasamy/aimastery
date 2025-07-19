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
            "content": "What is knowledge graph?"
        }
    ],
)

print(completion.choices[0].message)

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B:groq",
    messages=[
        {
            "role": "user",
            "content": "Explain graph database in 5 bullet points."
        }
    ],
)

print(completion.choices[0].message)

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    messages=[
        {
            "role": "user",
            "content": "Explain the difference between encode only transformer and decoder only transformer."
        }
    ],
)

print(completion.choices[0].message)
