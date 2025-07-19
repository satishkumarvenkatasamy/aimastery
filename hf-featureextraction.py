import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="sambanova",
    api_key=os.environ["HF_TOKEN"],
)

result = client.feature_extraction(
    "Today is a sunny day and I will get some ice cream.",
    model="intfloat/multilingual-e5-large",
)
