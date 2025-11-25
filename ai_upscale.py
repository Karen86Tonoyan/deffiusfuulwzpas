"""
AI Upscale - Zwiększanie rozdzielczości do 4K/8K
"""
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image, ImageEnhance
import torch

def upscale_4x(image_path: str, method="ai", output="upscaled_4x.png"):
    """
    Zwiększ rozdzielczość 4x

    Args:
        image_path: Ścieżka do obrazu (lub PIL.Image)
        method: 'ai' (wolniejsze, lepsze) lub 'fast' (szybkie)
        output: Gdzie zapisać

    Returns:
        str: Ścieżka do wyniku
    """
    print(f"Upscaling 4x (metoda: {method})...")

    # Wczytaj
    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    else:
        img = image_path.convert("RGB")

    print(f"Rozmiar wejściowy: {img.size}")

    if method == "ai":
        # AI Upscaling (Stable Diffusion)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=dtype
        )
        pipe = pipe.to(device)

        if device == "cuda":
            pipe.enable_attention_slicing()

        result = pipe(
            prompt="high quality, detailed, sharp, professional",
            image=img,
            num_inference_steps=50
        ).images[0]

    else:
        # Szybki upscaling (Lanczos)
        new_size = (img.width * 4, img.height * 4)
        result = img.resize(new_size, Image.Resampling.LANCZOS)

        # Popraw ostrość
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(1.3)

    print(f"Rozmiar wyjściowy: {result.size}")

    result.save(output)
    print(f"Zapisano: {output}")

    return output


def upscale_to_4k(image_path: str, output="upscaled_4k.png"):
    """Upscale do 4K (3840x2160)"""
    print("Upscaling do 4K...")

    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    else:
        img = image_path.convert("RGB")

    # Najpierw AI upscale
    upscaled = upscale_4x(img, method="ai", output="temp.png")

    # Potem resize do dokładnie 4K
    img_up = Image.open(upscaled)
    img_up.thumbnail((3840, 2160), Image.Resampling.LANCZOS)

    img_up.save(output)
    print(f"Zapisano 4K: {output}")

    return output


def enhance_photo(image_path: str, sharpness=1.3, color=1.2,
                 contrast=1.1, brightness=1.0, output="enhanced.png"):
    """
    Szybka poprawa jakości zdjęcia

    Args:
        image_path: Ścieżka do zdjęcia
        sharpness: Ostrość (1.0 = bez zmian)
        color: Saturacja
        contrast: Kontrast
        brightness: Jasność
        output: Gdzie zapisać
    """
    print("Poprawa jakości...")

    img = Image.open(image_path)

    # Ostrość
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(sharpness)

    # Kolory
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(color)

    # Kontrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)

    # Jasność
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness)

    img.save(output)
    print(f"Zapisano: {output}")

    return output


if __name__ == "__main__":
    print("Przykład użycia:")
    print('upscale_4x("photo.jpg", method="ai")')
    print('upscale_to_4k("photo.jpg")')
    print('enhance_photo("photo.jpg")')
