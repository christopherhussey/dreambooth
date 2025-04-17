import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def predict(prompt: str) -> Image.Image:
    model_path = "./weights/cgh-test-model.ckpt"

    # Load the pipeline
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate the image
    image = pipe(prompt).images[0]
    return image
