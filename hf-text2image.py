import os
from PIL import Image
from huggingface_hub import InferenceClient

'''
    black-forest-labs/FLUX.1-dev: One of the most powerful image generation models that can generate realistic outputs.
    latent-consistency/lcm-lora-sdxl: A powerful yet fast image generation model.
    Kwai-Kolors/Kolors: Text-to-image model for photorealistic generation.
    stabilityai/stable-diffusion-3-medium-diffusers: A powerful text-to-image model.
'''
models = [ "black-forest-labs/FLUX.1-dev",
           "latent-consistency/lcm-lora-sdxl",
           "Kwai-Kolors/Kolors",
           "stabilityai/stable-diffusion-3-medium-diffusers" ]

client = InferenceClient(
    provider="nebius",
    api_key=os.environ["HF_TOKEN"],
)

# output is a PIL.Image object
image = client.text_to_image(
    "An traiditional tamil girl near mettur dam",
    model="black-forest-labs/FLUX.1-dev",
#     model=models[3]
)

image.save("hf-t2i.jpg", "JPEG")
