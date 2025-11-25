"""
👔 WIRTUALNA PRZYMIERZALNIA - Zmień ubrania na zdjęciu!
========================================================

Funkcje:
- Zmiana ubrań (koszulka, sukienka, garnitur)
- Zmiana koloru ubrań
- Dodawanie akcesoriów
- Zmiana fryzury
- Pełna transformacja stylu
"""

from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import torch
import os

class VirtualDressingRoom:
    def __init__(self):
        """Inicjalizuj wirtualną przymierzalnię"""
        print("👔 Wirtualna Przymierzalnia - Inicjalizacja...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"Urządzenie: {self.device}")
        print("Ładowanie modelu inpainting...")

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=self.dtype,
            use_safetensors=True
        )
        self.pipe = self.pipe.to(self.device)

        if self.device == "cuda":
            self.pipe.enable_attention_slicing()

        print("✅ Gotowe!")

    def change_clothes(self, image_path, mask_path, prompt,
                      guidance=7.5, strength=0.75, output_path="result.png"):
        """
        Zmień ubrania na zdjęciu

        Args:
            image_path: Ścieżka do zdjęcia osoby
            mask_path: Ścieżka do maski (białe = zmień, czarne = zostaw)
            prompt: Opis nowych ubrań (po angielsku)
            guidance: Siła zgodności z promptem (5-15, wyżej = bardziej zgodny)
            strength: Siła zmiany (0.5-1.0, wyżej = więcej zmian)
            output_path: Gdzie zapisać wynik

        Returns:
            PIL.Image: Wynikowy obraz
        """
        print(f"\n👔 Zmiana ubrań...")
        print(f"Prompt: {prompt}")
        print(f"Guidance: {guidance}, Strength: {strength}")

        # Załaduj obrazy
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # Zmień rozmiar do 512x512 (optymalne dla SD)
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))

        # Generuj
        result = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            guidance_scale=guidance,
            strength=strength,
            num_inference_steps=50
        ).images[0]

        # Zapisz
        result.save(output_path)
        print(f"✅ Zapisano: {output_path}")

        return result

    def auto_create_clothing_mask(self, image_path, output_mask="auto_mask.png"):
        """
        Pomocnik: Stwórz prostą maskę dla górnej części ciała
        (Prosta wersja - w Paint będzie lepiej, ale to szybki start)

        Args:
            image_path: Ścieżka do zdjęcia
            output_mask: Gdzie zapisać maskę
        """
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Stwórz maskę - prostokąt w środkowej części (gdzie zwykle ubrania)
        mask = Image.new('RGB', (width, height), 'black')
        draw = ImageDraw.Draw(mask)

        # Górna część ciała (w przybliżeniu)
        top = int(height * 0.2)    # Od 20% wysokości
        bottom = int(height * 0.7)  # Do 70% wysokości
        left = int(width * 0.2)     # Od 20% szerokości
        right = int(width * 0.8)    # Do 80% szerokości

        # Rysuj białą elipsę (przybliżony kształt tułowia)
        draw.ellipse([left, top, right, bottom], fill='white')

        mask.save(output_mask)
        print(f"✅ Utworzono prostą maskę: {output_mask}")
        print(f"UWAGA: To prosta maska! Dla lepszych wyników:")
        print(f"  1. Otwórz {output_mask} w Paint")
        print(f"  2. Dokładnie zamaluj NA BIAŁO tylko ubrania")
        print(f"  3. Zapisz i użyj ponownie")

        return mask


# ============================================================================
#                        PRZYKŁADY UŻYCIA
# ============================================================================

def example_change_shirt():
    """Przykład: Zmień koszulkę"""
    room = VirtualDressingRoom()

    # UWAGA: Najpierw musisz mieć:
    # 1. selfie.jpg - zdjęcie osoby
    # 2. mask.png - maska (białe = ubranie, czarne = reszta)

    room.change_clothes(
        image_path="selfie.jpg",
        mask_path="mask.png",
        prompt="red t-shirt, casual, cotton, high quality",
        guidance=7.5,
        strength=0.75,
        output_path="czerwona_koszulka.png"
    )


def example_formal_outfit():
    """Przykład: Elegancki strój"""
    room = VirtualDressingRoom()

    room.change_clothes(
        image_path="selfie.jpg",
        mask_path="mask.png",
        prompt="elegant black suit, formal, professional, businessman",
        guidance=8.0,
        strength=0.8,
        output_path="garnitur.png"
    )


def example_dress():
    """Przykład: Sukienka"""
    room = VirtualDressingRoom()

    room.change_clothes(
        image_path="selfie.jpg",
        mask_path="mask.png",
        prompt="beautiful blue dress, elegant, evening gown, luxurious",
        guidance=7.5,
        strength=0.8,
        output_path="sukienka.png"
    )


def example_change_color():
    """Przykład: Zmień tylko kolor"""
    room = VirtualDressingRoom()

    room.change_clothes(
        image_path="selfie.jpg",
        mask_path="mask.png",
        prompt="same style but pink color, vibrant, fashionable",
        guidance=6.0,
        strength=0.5,  # Niższa siła = mniej zmian
        output_path="rozowy_kolor.png"
    )


# ============================================================================
#                        INTERAKTYWNY TRYB
# ============================================================================

def interactive_mode():
    """Interaktywny tryb - wybieraj opcje menu"""
    print("=" * 70)
    print("           👔 WIRTUALNA PRZYMIERZALNIA")
    print("=" * 70)
    print()

    # Inicjalizuj
    room = VirtualDressingRoom()

    while True:
        print("\n" + "=" * 70)
        print("MENU GŁÓWNE:")
        print("=" * 70)
        print()
        print("1. 👕 Zmień koszulkę/bluzę")
        print("2. 👔 Dodaj garnitur/marynarkę")
        print("3. 👗 Zmień na sukienkę")
        print("4. 🎨 Zmień tylko kolor ubrania")
        print("5. 👔 Własny prompt (zaawansowane)")
        print("6. 🎭 Stwórz prostą maskę (auto)")
        print("0. ❌ Wyjście")
        print()

        choice = input("Wybierz opcję (0-6): ").strip()

        if choice == "0":
            print("\nDo zobaczenia! 👋")
            break

        elif choice == "6":
            # Auto-maska
            image_path = input("\nŚcieżka do zdjęcia: ").strip()
            if not os.path.exists(image_path):
                print(f"❌ Plik nie istnieje: {image_path}")
                continue

            mask_path = input("Gdzie zapisać maskę? (Enter = auto_mask.png): ").strip()
            if not mask_path:
                mask_path = "auto_mask.png"

            room.auto_create_clothing_mask(image_path, mask_path)
            print(f"\n✅ Maska zapisana: {mask_path}")
            print("Teraz możesz użyć opcji 1-5 z tą maską!")
            continue

        # Dla opcji 1-5 potrzebujemy zdjęcia i maski
        print()
        image_path = input("Ścieżka do zdjęcia: ").strip()
        if not os.path.exists(image_path):
            print(f"❌ Plik nie istnieje: {image_path}")
            continue

        mask_path = input("Ścieżka do maski: ").strip()
        if not os.path.exists(mask_path):
            print(f"❌ Plik nie istnieje: {mask_path}")
            continue

        output_path = input("Nazwa pliku wynikowego (Enter = result.png): ").strip()
        if not output_path:
            output_path = "result.png"

        # Wykonaj operację
        if choice == "1":
            # Koszulka
            print("\nJaki styl koszulki?")
            print("  a) Biała, klasyczna")
            print("  b) Czarna, sportowa")
            print("  c) Kolorowa, casual")
            print("  d) Własny opis")

            style = input("Wybierz (a-d): ").strip().lower()

            prompts = {
                'a': "white t-shirt, classic, cotton, casual, high quality",
                'b': "black sports t-shirt, athletic, modern, fitness style",
                'c': "colorful casual t-shirt, vibrant, trendy, fashionable",
            }

            if style in prompts:
                prompt = prompts[style]
            else:
                prompt = input("Opisz koszulkę (po angielsku): ").strip()

            room.change_clothes(image_path, mask_path, prompt,
                              guidance=7.5, strength=0.75, output_path=output_path)

        elif choice == "2":
            # Garnitur
            print("\nJaki styl?")
            print("  a) Czarny garnitur (biznesowy)")
            print("  b) Granatowy garnitur (elegancki)")
            print("  c) Marynarka casual")

            style = input("Wybierz (a-c): ").strip().lower()

            prompts = {
                'a': "black formal suit, business, professional, elegant",
                'b': "navy blue suit, elegant, formal, high quality",
                'c': "casual blazer, smart casual, modern, stylish",
            }

            prompt = prompts.get(style, "formal suit, elegant, professional")

            room.change_clothes(image_path, mask_path, prompt,
                              guidance=8.0, strength=0.8, output_path=output_path)

        elif choice == "3":
            # Sukienka
            print("\nJaki styl sukienki?")
            print("  a) Elegancka wieczorowa")
            print("  b) Letnia, kolorowa")
            print("  c) Koktajlowa")

            style = input("Wybierz (a-c): ").strip().lower()

            prompts = {
                'a': "elegant evening gown, luxurious, formal, beautiful",
                'b': "summer dress, colorful, floral, casual, light",
                'c': "cocktail dress, stylish, party, modern, chic",
            }

            prompt = prompts.get(style, "beautiful dress, elegant, fashionable")

            room.change_clothes(image_path, mask_path, prompt,
                              guidance=7.5, strength=0.8, output_path=output_path)

        elif choice == "4":
            # Zmień kolor
            color = input("\nJaki kolor? (np. red, blue, green, pink): ").strip()
            prompt = f"same clothing style but {color} color, vibrant, high quality"

            room.change_clothes(image_path, mask_path, prompt,
                              guidance=6.0, strength=0.5, output_path=output_path)

        elif choice == "5":
            # Własny prompt
            prompt = input("\nOpisz ubranie (po angielsku): ").strip()

            print("\nParametry zaawansowane (Enter = domyślne):")
            guidance_input = input("Guidance (5-15, domyślnie 7.5): ").strip()
            strength_input = input("Strength (0.1-1.0, domyślnie 0.75): ").strip()

            guidance = float(guidance_input) if guidance_input else 7.5
            strength = float(strength_input) if strength_input else 0.75

            room.change_clothes(image_path, mask_path, prompt,
                              guidance=guidance, strength=strength,
                              output_path=output_path)

        print(f"\n✅ Gotowe! Sprawdź: {output_path}")

        again = input("\nKolejne przymierzanie? (t/n): ").strip().lower()
        if again != 't':
            print("\nDo zobaczenia! 👋")
            break


# ============================================================================
#                        PRESETS - GOTOWE STYLE
# ============================================================================

CLOTHING_PRESETS = {
    # Męskie
    "m_casual": "casual t-shirt and jeans, modern, comfortable, everyday style",
    "m_formal": "black formal suit, white shirt, tie, business, professional",
    "m_sport": "athletic sportswear, gym outfit, modern, fitness",
    "m_summer": "light summer shirt, shorts, casual, beach style",
    "m_smart": "smart casual, blazer, dress shirt, stylish, modern",

    # Damskie
    "f_casual": "casual blouse and jeans, modern, comfortable, everyday",
    "f_formal": "elegant business suit, professional, formal, chic",
    "f_dress": "beautiful cocktail dress, elegant, party, stylish",
    "f_summer": "light summer dress, colorful, floral, casual",
    "f_sport": "athletic sportswear, yoga outfit, modern, fitness",

    # Kolory
    "red": "same style but red color, vibrant, fashionable",
    "blue": "same style but blue color, elegant, professional",
    "black": "same style but black color, classic, sophisticated",
    "white": "same style but white color, clean, fresh",
    "pink": "same style but pink color, vibrant, trendy",
}

def use_preset(preset_name):
    """Użyj gotowego presetu"""
    if preset_name not in CLOTHING_PRESETS:
        print(f"❌ Nieznany preset: {preset_name}")
        print(f"Dostępne: {', '.join(CLOTHING_PRESETS.keys())}")
        return

    room = VirtualDressingRoom()

    image_path = input("Ścieżka do zdjęcia: ").strip()
    mask_path = input("Ścieżka do maski: ").strip()

    prompt = CLOTHING_PRESETS[preset_name]
    print(f"\n📝 Używam presetu '{preset_name}': {prompt}")

    room.change_clothes(
        image_path=image_path,
        mask_path=mask_path,
        prompt=prompt,
        output_path=f"preset_{preset_name}.png"
    )


# ============================================================================
#                           URUCHOM PROGRAM
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║           👔 WIRTUALNA PRZYMIERZALNIA AI 👗                      ║
    ║                                                                  ║
    ║  Zmień ubrania na zdjęciu bez wychodzenia z domu!                ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝

    JAK UŻYWAĆ:

    1. PRZYGOTUJ ZDJĘCIE
       - Zrób selfie lub użyj istniejącego zdjęcia
       - Najlepiej na prostym tle
       - Osoba od pasa w górę lub w całości

    2. STWÓRZ MASKĘ
       Opcja A (Auto):
         → Wybierz opcję 6 w menu
         → Program stworzy prostą maskę
         → Popraw ją w Paint jeśli trzeba

       Opcja B (Ręcznie):
         → Otwórz zdjęcie w Paint
         → Zamaluj NA BIAŁO tylko ubranie (to co chcesz zmienić)
         → Reszta niech będzie CZARNA
         → Zapisz jako mask.png

    3. WYBIERZ STYL
       → Uruchom program
       → Wybierz z menu (koszulka, garnitur, sukienka...)
       → Poczekaj na wynik
       → Gotowe!

    ══════════════════════════════════════════════════════════════════

    PRZYKŁADY:

    • Zamień zwykłą koszulkę na garnitur (rozmowa o pracę!)
    • Zobacz jak będziesz wyglądać w różnych kolorach
    • Przymierz sukienkę przed zakupem
    • Dodaj marynarkę do casualowego zdjęcia

    ══════════════════════════════════════════════════════════════════
    """)

    input("Naciśnij Enter żeby rozpocząć...")

    # Uruchom interaktywny tryb
    interactive_mode()
