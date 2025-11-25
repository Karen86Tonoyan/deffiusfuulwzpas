"""
AI Text PL - Dodawanie polskiego tekstu do obrazów
"""
from PIL import Image, ImageDraw, ImageFont

def add_polish_text(image_path, text, position=(40, 40), font_size=64,
                   color=(255, 255, 255), font_path="arial.ttf",
                   shadow=True, output="with_text.png"):
    """
    Dodaj polski tekst do obrazu

    Args:
        image_path: Ścieżka do obrazu (lub PIL.Image)
        text: Tekst (POLSKIE ZNAKI OK!)
        position: Pozycja (x, y)
        font_size: Rozmiar czcionki
        color: Kolor RGB tuple
        font_path: Ścieżka do czcionki TTF
        shadow: Czy dodać cień
        output: Gdzie zapisać

    Returns:
        str: Ścieżka do wyniku
    """
    print(f"Dodawanie tekstu: '{text}'")

    # Wczytaj obraz
    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    else:
        img = image_path.convert("RGB")

    draw = ImageDraw.Draw(img)

    # Załaduj czcionkę
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        print(f"⚠️  Nie znaleziono {font_path}, używam domyślnej")
        font = ImageFont.load_default()

    # Cień (dla lepszej czytelności)
    if shadow:
        shadow_offset = max(2, font_size // 20)
        shadow_pos = (position[0] + shadow_offset, position[1] + shadow_offset)
        draw.text(shadow_pos, text, font=font, fill=(0, 0, 0))

    # Główny tekst
    draw.text(position, text, font=font, fill=color)

    img.save(output)
    print(f"Zapisano: {output}")

    return output


def add_watermark(image_path, watermark_text, position='bottom-right',
                 font_size=24, opacity=0.5, output="watermarked.png"):
    """
    Dodaj watermark (znak wodny)

    Args:
        image_path: Ścieżka do obrazu
        watermark_text: Tekst watermarku
        position: 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'
        font_size: Rozmiar czcionki
        opacity: Przezroczystość (0.0-1.0)
        output: Gdzie zapisać
    """
    # Wczytaj
    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    else:
        img = image_path.convert("RGB")

    # Warstwa watermarku
    watermark = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)

    # Czcionka
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Pozycja
    bbox = draw.textbbox((0, 0), watermark_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    margin = 20

    if position == 'top-left':
        pos = (margin, margin)
    elif position == 'top-right':
        pos = (img.width - text_width - margin, margin)
    elif position == 'bottom-left':
        pos = (margin, img.height - text_height - margin)
    elif position == 'bottom-right':
        pos = (img.width - text_width - margin, img.height - text_height - margin)
    else:  # center
        pos = ((img.width - text_width) // 2, (img.height - text_height) // 2)

    # Rysuj z przezroczystością
    alpha = int(255 * opacity)
    draw.text(pos, watermark_text, font=font, fill=(255, 255, 255, alpha))

    # Nałóż
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, watermark)
    img = img.convert('RGB')

    img.save(output)
    print(f"Watermark dodany: {output}")

    return output


if __name__ == "__main__":
    # Test
    # add_polish_text(
    #     "photo.jpg",
    #     "POLSKIE ZNAKI: ąćęłńóśźż ĄĆĘŁŃÓŚŹŻ",
    #     position=(50, 50),
    #     font_size=72
    # )
    print("Przykład użycia:")
    print('add_polish_text("photo.jpg", "Mój Tekst PL", position=(50,50))')
