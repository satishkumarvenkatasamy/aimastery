import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="groq",
    api_key=os.environ["HF_TOKEN"],
)

print("-"*50)
print("Asking meta-llama/Llama-3.3-70B-Instruct this question:")
print("Explain graph database")
print("-"*50)
completion = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Explain graph database"
        }
    ],
)
print(completion.choices[0].message.content)


print("-"*50)
print("Asking deepseek-ai/DeepSeek-R1-Distill-Llama-70B to answer this question:")
print("Explain no-sql database. What are the different types of nosql databases?")
print("-"*50)
completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    messages=[
        {
            "role": "user",
            "content": "Explain no-sql database. What are the different types of nosql databases?"
        }
    ],
)

print(completion.choices[0].message.content)

print("-"*50)
print("Asking deepseek-ai/DeepSeek-R1-Distill-Llama-70B to answer this question:")
print("What is decoder-only transformer?")
print("-"*50)
completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    messages=[
        {
            "role": "user",
            "content": "What is decoder-only transformer?"
        }
    ],
)

print(completion.choices[0].message.content)
