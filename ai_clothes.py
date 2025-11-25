"""
AI Clothes - Zmiana ubrań na zdjęciu
"""
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

def change_clothes(image_path: str, mask_path: str, prompt: str,
                  guidance=7.5, steps=35, output="clothes_changed.png"):
    """
    Zmień ubrania na zdjęciu

    Args:
        image_path: Ścieżka do zdjęcia (lub PIL.Image)
        mask_path: Ścieżka do maski (białe = zmień, czarne = zostaw)
        prompt: Opis nowych ubrań
        guidance: Siła promptu (5-15)
        steps: Kroki generowania (25-50)
        output: Gdzie zapisać

    Returns:
        str: Ścieżka do wyniku
    """
    print(f"Zmiana ubrań...")
    print(f"Prompt: {prompt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Załaduj model inpainting
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=dtype,
        use_safetensors=True
    )
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    # Wczytaj obrazy
    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    else:
        img = image_path.convert("RGB")

    if isinstance(mask_path, str):
        mask = Image.open(mask_path).convert("RGB")
    else:
        mask = mask_path.convert("RGB")

    # Zmień rozmiar do 512x512 (optymalne)
    img = img.resize((512, 512))
    mask = mask.resize((512, 512))

    # Generuj
    result = pipe(
        prompt=prompt,
        image=img,
        mask_image=mask,
        num_inference_steps=steps,
        guidance_scale=guidance
    ).images[0]

    result.save(output)
    print(f"Zapisano: {output}")

    return output


if __name__ == "__main__":
    # Test (potrzebujesz selfie.jpg i mask.png)
    # change_clothes(
    #     "selfie.jpg",
    #     "mask.png",
    #     "red t-shirt, casual, high quality"
    # )
    print("Przykład użycia:")
    print('change_clothes("selfie.jpg", "mask.png", "red t-shirt")')
