import os
import torch
import requests
from diffusers import StableDiffusionPipeline
from PIL import Image

MODEL_PATH = "cgh-test-model.ckpt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1T0lPBwutGIuJMQypBLKpdcBaXDGtSGEh"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

def predict(prompt: str) -> Image.Image:
    download_model()

    pipe = StableDiffusionPipeline.from_single_file(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    image = pipe(prompt).images[0]
    return image
