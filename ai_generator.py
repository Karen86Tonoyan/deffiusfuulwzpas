"""
AI Generator - Generowanie obrazów z tekstu
"""
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image
import torch

def generate_image(prompt: str, negative_prompt="", res="1024x1024",
                  model="sdxl", output="generated.png"):
    """
    Generuj obraz z promptu

    Args:
        prompt: Opis sceny
        negative_prompt: Czego unikać
        res: Rozdzielczość "WIDTHxHEIGHT"
        model: 'sdxl' (najlepszy), 'sd21', 'sd15'
        output: Ścieżka zapisu

    Returns:
        str: Ścieżka do wygenerowanego obrazu
    """
    w, h = map(int, res.split("x"))

    print(f"Generowanie: {w}x{h}")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Wybierz model
    if model == "sdxl":
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True
        )
    elif model == "sd21":
        model_id = "stabilityai/stable-diffusion-2-1"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True
        )
    else:  # sd15
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True
        )

    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    # Generuj
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=w,
        height=h,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]

    result.save(output)
    print(f"Zapisano: {output}")

    return output


if __name__ == "__main__":
    # Test
    generate_image(
        prompt="beautiful landscape, mountains, sunset, 4k, photorealistic",
        res="1024x1024",
        model="sdxl"
    )
