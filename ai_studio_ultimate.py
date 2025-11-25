"""
🎨 AI STUDIO ULTIMATE - Pełna wersja z wszystkim!
=================================================

✅ Checkpointy (własne modele .safetensors)
✅ VAE (lepsze kolory)
✅ LoRA (dodatkowe style)
✅ Dodawanie POLSKIEGO TEKSTU do obrazów
✅ Wszystkie formaty: PNG, JPG, TIFF, WEBP
✅ GUI + wszystkie funkcje
"""

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    AutoencoderKL
)
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import torch
import os
from pathlib import Path

class AIStudioUltimate:
    def __init__(self):
        """Inicjalizacja Ultimate Studio"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.pipe = None
        self.current_checkpoint = None
        self.current_vae = None
        self.loaded_loras = []

        print("🎨 AI Studio Ultimate - Inicjalizacja")
        print(f"Urządzenie: {self.device}")

    # ========================================================================
    #                     ZARZĄDZANIE MODELAMI
    # ========================================================================

    def load_checkpoint(self, checkpoint_path_or_id, use_safetensors=True):
        """
        Załaduj checkpoint (model)

        Args:
            checkpoint_path_or_id: Ścieżka do .safetensors LUB ID z HuggingFace
            use_safetensors: Czy używać safetensors (bezpieczniejsze)

        Przykłady:
            # Z HuggingFace:
            load_checkpoint("stabilityai/stable-diffusion-2-1")

            # Lokalny plik:
            load_checkpoint("models/my_model.safetensors")
        """
        print(f"\n📥 Ładowanie checkpointu: {checkpoint_path_or_id}")

        # Sprawdź czy to lokalny plik czy HF repo
        if os.path.exists(checkpoint_path_or_id):
            # Lokalny plik .safetensors lub .ckpt
            print("Ładowanie z lokalnego pliku...")

            # Dla single file checkpoints
            from diffusers import StableDiffusionPipeline

            if checkpoint_path_or_id.endswith('.safetensors'):
                self.pipe = StableDiffusionPipeline.from_single_file(
                    checkpoint_path_or_id,
                    torch_dtype=self.dtype,
                    use_safetensors=True
                )
            else:
                self.pipe = StableDiffusionPipeline.from_single_file(
                    checkpoint_path_or_id,
                    torch_dtype=self.dtype
                )
        else:
            # HuggingFace repo
            print("Pobieranie z HuggingFace...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                checkpoint_path_or_id,
                torch_dtype=self.dtype,
                use_safetensors=use_safetensors
            )

        self.pipe = self.pipe.to(self.device)

        if self.device == "cuda":
            self.pipe.enable_attention_slicing()

        self.current_checkpoint = checkpoint_path_or_id
        print(f"✅ Checkpoint załadowany: {checkpoint_path_or_id}")

    def load_vae(self, vae_path_or_id):
        """
        Załaduj VAE (poprawia kolory i szczegóły)

        Args:
            vae_path_or_id: Ścieżka do VAE lub HF repo

        Przykłady:
            # Oficjalny VAE dla SD 2.1:
            load_vae("stabilityai/sd-vae-ft-mse")

            # Lokalny:
            load_vae("models/vae.safetensors")
        """
        print(f"\n📥 Ładowanie VAE: {vae_path_or_id}")

        if not self.pipe:
            print("❌ Najpierw załaduj checkpoint!")
            return

        if os.path.exists(vae_path_or_id):
            # Lokalny plik
            vae = AutoencoderKL.from_single_file(
                vae_path_or_id,
                torch_dtype=self.dtype
            )
        else:
            # HuggingFace
            vae = AutoencoderKL.from_pretrained(
                vae_path_or_id,
                torch_dtype=self.dtype
            )

        vae = vae.to(self.device)
        self.pipe.vae = vae

        self.current_vae = vae_path_or_id
        print(f"✅ VAE załadowany: {vae_path_or_id}")

    def load_lora(self, lora_path, weight=1.0, adapter_name=None):
        """
        Załaduj LoRA (dodatkowe style)

        Args:
            lora_path: Ścieżka do pliku .safetensors LoRA
            weight: Siła LoRA (0.0-1.5, domyślnie 1.0)
            adapter_name: Nazwa adaptera (opcjonalnie)

        Przykłady:
            load_lora("models/anime_style.safetensors", weight=0.8)
            load_lora("models/realistic.safetensors", weight=1.2)
        """
        print(f"\n📥 Ładowanie LoRA: {lora_path} (waga: {weight})")

        if not self.pipe:
            print("❌ Najpierw załaduj checkpoint!")
            return

        if not adapter_name:
            adapter_name = Path(lora_path).stem  # Nazwa pliku bez rozszerzenia

        # Załaduj LoRA
        self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)

        # Ustaw wagę
        self.pipe.set_adapters([adapter_name], adapter_weights=[weight])

        self.loaded_loras.append({
            'name': adapter_name,
            'path': lora_path,
            'weight': weight
        })

        print(f"✅ LoRA załadowany: {adapter_name}")

    def unload_lora(self, adapter_name=None):
        """Wyładuj LoRA"""
        if adapter_name:
            self.pipe.delete_adapters(adapter_name)
            self.loaded_loras = [l for l in self.loaded_loras if l['name'] != adapter_name]
            print(f"✅ LoRA wyładowany: {adapter_name}")
        else:
            # Wyładuj wszystkie
            self.pipe.unload_lora_weights()
            self.loaded_loras = []
            print("✅ Wszystkie LoRA wyładowane")

    def list_loaded_models(self):
        """Wyświetl załadowane modele"""
        print("\n" + "="*60)
        print("ZAŁADOWANE MODELE:")
        print("="*60)
        print(f"Checkpoint: {self.current_checkpoint or 'Brak'}")
        print(f"VAE: {self.current_vae or 'Domyślny'}")
        print(f"LoRA: {len(self.loaded_loras)} załadowanych")
        for lora in self.loaded_loras:
            print(f"  • {lora['name']} (waga: {lora['weight']})")
        print("="*60)

    # ========================================================================
    #                     DODAWANIE TEKSTU (POLSKIE ZNAKI!)
    # ========================================================================

    def add_text_to_image(self, image_path, text, position=(50, 50),
                         font_size=48, color=(255, 255, 255),
                         font_path="arial.ttf", output_format="png",
                         output_path=None):
        """
        Dodaj POLSKI TEKST do obrazu

        Args:
            image_path: Ścieżka do obrazu (str lub PIL.Image)
            text: Tekst do dodania (POLSKIE ZNAKI OK!)
            position: Pozycja (x, y) tuple
            font_size: Rozmiar czcionki
            color: Kolor tekstu (R, G, B) tuple
            font_path: Ścieżka do czcionki TTF
            output_format: Format wyjściowy (png, jpg, tiff, webp)
            output_path: Gdzie zapisać (None = auto)

        Returns:
            PIL.Image: Obraz z tekstem
        """
        print(f"\n✍️  Dodawanie tekstu: '{text}'")

        # Wczytaj obraz
        if isinstance(image_path, str):
            img = Image.open(image_path).convert("RGB")
        else:
            img = image_path.convert("RGB")

        # Rysuj
        draw = ImageDraw.Draw(img)

        # Załaduj czcionkę
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            print(f"⚠️  Nie znaleziono czcionki {font_path}, używam domyślnej")
            font = ImageFont.load_default()

        # Dodaj cień (dla lepszej czytelności)
        shadow_offset = max(2, font_size // 24)
        shadow_color = (0, 0, 0)

        # Cień
        draw.text(
            (position[0] + shadow_offset, position[1] + shadow_offset),
            text,
            font=font,
            fill=shadow_color
        )

        # Główny tekst
        draw.text(position, text, font=font, fill=color)

        # Zapisz
        if not output_path:
            base = Path(image_path).stem if isinstance(image_path, str) else "image"
            output_path = f"{base}_with_text.{output_format}"

        # Zapisz w odpowiednim formacie
        if output_format.lower() in ['jpg', 'jpeg']:
            img.save(output_path, format='JPEG', quality=95)
        elif output_format.lower() == 'webp':
            img.save(output_path, format='WEBP', quality=95)
        elif output_format.lower() == 'tiff':
            img.save(output_path, format='TIFF', compression='tiff_deflate')
        else:  # PNG
            img.save(output_path, format='PNG')

        print(f"✅ Zapisano: {output_path}")
        return img

    def add_watermark(self, image_path, watermark_text,
                     position='bottom-right', font_size=24,
                     opacity=0.5, output_path=None):
        """
        Dodaj watermark (znak wodny)

        Args:
            image_path: Ścieżka do obrazu
            watermark_text: Tekst watermarku
            position: Pozycja ('top-left', 'top-right', 'bottom-left', 'bottom-right', 'center')
            font_size: Rozmiar czcionki
            opacity: Przezroczystość (0.0-1.0)
            output_path: Gdzie zapisać
        """
        # Wczytaj obraz
        if isinstance(image_path, str):
            img = Image.open(image_path).convert("RGB")
        else:
            img = image_path.convert("RGB")

        # Stwórz warstwę dla watermarku
        watermark = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)

        # Czcionka
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Oblicz pozycję
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

        # Rysuj watermark z przezroczystością
        alpha = int(255 * opacity)
        draw.text(pos, watermark_text, font=font, fill=(255, 255, 255, alpha))

        # Nałóż watermark na obraz
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, watermark)
        img = img.convert('RGB')

        # Zapisz
        if not output_path:
            base = Path(image_path).stem if isinstance(image_path, str) else "image"
            output_path = f"{base}_watermarked.png"

        img.save(output_path)
        print(f"✅ Watermark dodany: {output_path}")
        return img

    # ========================================================================
    #                     KONWERSJA FORMATÓW
    # ========================================================================

    def convert_format(self, image_path, output_format, quality=95, output_path=None):
        """
        Konwertuj format obrazu

        Args:
            image_path: Ścieżka do obrazu
            output_format: Docelowy format (png, jpg, tiff, webp)
            quality: Jakość (1-100, dla JPG/WEBP)
            output_path: Gdzie zapisać

        Supported: PNG, JPG/JPEG, TIFF, WEBP, BMP, GIF
        """
        print(f"\n🔄 Konwersja: {image_path} → {output_format.upper()}")

        img = Image.open(image_path)

        if not output_path:
            base = Path(image_path).stem
            output_path = f"{base}.{output_format}"

        # Konwertuj do RGB jeśli potrzeba
        if output_format.lower() in ['jpg', 'jpeg'] and img.mode != 'RGB':
            img = img.convert('RGB')

        # Zapisz w odpowiednim formacie
        save_kwargs = {}

        if output_format.lower() in ['jpg', 'jpeg']:
            save_kwargs = {'format': 'JPEG', 'quality': quality, 'optimize': True}
        elif output_format.lower() == 'webp':
            save_kwargs = {'format': 'WEBP', 'quality': quality}
        elif output_format.lower() == 'tiff':
            save_kwargs = {'format': 'TIFF', 'compression': 'tiff_deflate'}
        elif output_format.lower() == 'png':
            save_kwargs = {'format': 'PNG', 'optimize': True}
        elif output_format.lower() == 'bmp':
            save_kwargs = {'format': 'BMP'}
        elif output_format.lower() == 'gif':
            save_kwargs = {'format': 'GIF'}

        img.save(output_path, **save_kwargs)
        print(f"✅ Zapisano: {output_path}")

        # Pokaż rozmiar pliku
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"Rozmiar: {file_size:.1f} KB")

        return output_path

    def batch_convert(self, folder_path, output_format, quality=95):
        """
        Konwertuj wszystkie obrazy w folderze

        Args:
            folder_path: Ścieżka do folderu
            output_format: Format docelowy
            quality: Jakość
        """
        supported = ['.png', '.jpg', '.jpeg', '.tiff', '.webp', '.bmp', '.gif']

        folder = Path(folder_path)
        output_folder = folder / f"converted_{output_format}"
        output_folder.mkdir(exist_ok=True)

        images = []
        for ext in supported:
            images.extend(folder.glob(f"*{ext}"))

        print(f"\n🔄 Konwersja {len(images)} obrazów do {output_format.upper()}...")

        for img_path in images:
            output_path = output_folder / f"{img_path.stem}.{output_format}"
            self.convert_format(str(img_path), output_format, quality, str(output_path))

        print(f"\n✅ Gotowe! Wszystkie obrazy w: {output_folder}")

    # ========================================================================
    #                     GENEROWANIE OBRAZÓW
    # ========================================================================

    def generate(self, prompt, negative_prompt="", width=512, height=512,
                steps=50, guidance=7.5, output_path="output.png",
                output_format="png"):
        """
        Generuj obraz (text2img)

        Args:
            prompt: Opis obrazu
            negative_prompt: Czego unikać
            width: Szerokość
            height: Wysokość
            steps: Liczba kroków
            guidance: Siła zgodności z promptem
            output_path: Gdzie zapisać
            output_format: Format (png, jpg, tiff, webp)
        """
        if not self.pipe:
            print("❌ Najpierw załaduj checkpoint!")
            return None

        print(f"\n🎨 Generowanie obrazu...")
        print(f"Prompt: {prompt}")

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance
        ).images[0]

        # Zapisz w odpowiednim formacie
        if output_format.lower() in ['jpg', 'jpeg']:
            result.save(output_path, format='JPEG', quality=95)
        elif output_format.lower() == 'webp':
            result.save(output_path, format='WEBP', quality=95)
        elif output_format.lower() == 'tiff':
            result.save(output_path, format='TIFF')
        else:
            result.save(output_path, format='PNG')

        print(f"✅ Zapisano: {output_path}")
        return result


# ============================================================================
#                        PRZYKŁADY UŻYCIA
# ============================================================================

def example_basic():
    """Podstawowe użycie"""
    studio = AIStudioUltimate()

    # Załaduj domyślny model
    studio.load_checkpoint("stabilityai/stable-diffusion-2-1")

    # Generuj
    studio.generate(
        prompt="beautiful landscape, mountains, sunset, 4k",
        output_path="landscape.png"
    )


def example_with_vae():
    """Z własnym VAE"""
    studio = AIStudioUltimate()

    # Załaduj checkpoint
    studio.load_checkpoint("stabilityai/stable-diffusion-2-1")

    # Załaduj lepszy VAE (poprawia kolory!)
    studio.load_vae("stabilityai/sd-vae-ft-mse")

    # Generuj
    studio.generate(
        prompt="portrait of a woman, professional photo, detailed",
        output_path="portrait_with_vae.png"
    )


def example_with_lora():
    """Z LoRA (dodatkowy styl)"""
    studio = AIStudioUltimate()

    studio.load_checkpoint("stabilityai/stable-diffusion-2-1")

    # Załaduj LoRA (np. styl anime)
    # studio.load_lora("models/anime_style.safetensors", weight=0.8)

    studio.generate(
        prompt="anime girl, colorful, detailed",
        output_path="anime_style.png"
    )


def example_add_polish_text():
    """Dodaj polski tekst"""
    studio = AIStudioUltimate()

    # Najpierw wygeneruj obraz
    studio.load_checkpoint("stabilityai/stable-diffusion-2-1")
    img = studio.generate(
        prompt="beautiful poster background, gradient, modern",
        output_path="base.png"
    )

    # Dodaj polski tekst
    studio.add_text_to_image(
        image_path="base.png",
        text="POLSKIE ZNAKI: ąćęłńóśźż",
        position=(50, 50),
        font_size=60,
        color=(255, 255, 255),
        output_format="png",
        output_path="z_polskim_tekstem.png"
    )


def example_convert_formats():
    """Konwersja formatów"""
    studio = AIStudioUltimate()

    # PNG → JPG
    studio.convert_format("image.png", "jpg", quality=90)

    # PNG → WEBP (mniejszy rozmiar!)
    studio.convert_format("image.png", "webp", quality=85)

    # PNG → TIFF (archiwalna jakość)
    studio.convert_format("image.png", "tiff")


def example_watermark():
    """Dodaj watermark"""
    studio = AIStudioUltimate()

    studio.add_watermark(
        image_path="photo.jpg",
        watermark_text="© 2025 Moja Firma",
        position='bottom-right',
        font_size=24,
        opacity=0.5,
        output_path="photo_watermark.png"
    )


# ============================================================================
#                     INTERAKTYWNE MENU
# ============================================================================

def interactive_menu():
    """Interaktywne menu"""
    studio = AIStudioUltimate()

    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║              🎨 AI STUDIO ULTIMATE 🎨                            ║
    ║                                                                  ║
    ║  ✅ Checkpointy  ✅ VAE  ✅ LoRA                                  ║
    ║  ✅ Polski tekst  ✅ Wszystkie formaty                           ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    while True:
        print("\n" + "="*70)
        print("MENU GŁÓWNE:")
        print("="*70)
        print()
        print("MODELE:")
        print("  1. 📥 Załaduj checkpoint")
        print("  2. 🎨 Załaduj VAE")
        print("  3. ✨ Załaduj LoRA")
        print("  4. 📋 Pokaż załadowane modele")
        print()
        print("GENEROWANIE:")
        print("  5. 🎨 Generuj obraz (text2img)")
        print()
        print("TEKST:")
        print("  6. ✍️  Dodaj tekst do obrazu (POLSKIE ZNAKI!)")
        print("  7. 💧 Dodaj watermark")
        print()
        print("FORMATY:")
        print("  8. 🔄 Konwertuj format")
        print("  9. 📁 Konwertuj folder (batch)")
        print()
        print("  0. ❌ Wyjście")
        print()

        choice = input("Wybierz (0-9): ").strip()

        if choice == "0":
            print("\nDo zobaczenia! 👋")
            break

        elif choice == "1":
            # Załaduj checkpoint
            print("\nOpcje:")
            print("  1) Stable Diffusion 2.1 (domyślny)")
            print("  2) Stable Diffusion 1.5")
            print("  3) Własny plik .safetensors")

            opt = input("Wybierz (1-3): ").strip()

            if opt == "1":
                studio.load_checkpoint("stabilityai/stable-diffusion-2-1")
            elif opt == "2":
                studio.load_checkpoint("runwayml/stable-diffusion-v1-5")
            elif opt == "3":
                path = input("Ścieżka do .safetensors: ").strip()
                if os.path.exists(path):
                    studio.load_checkpoint(path)
                else:
                    print("❌ Plik nie istnieje!")

        elif choice == "2":
            # Załaduj VAE
            print("\nOpcje:")
            print("  1) SD VAE ft-mse (lepsze kolory)")
            print("  2) Własny plik")

            opt = input("Wybierz (1-2): ").strip()

            if opt == "1":
                studio.load_vae("stabilityai/sd-vae-ft-mse")
            elif opt == "2":
                path = input("Ścieżka do VAE: ").strip()
                if os.path.exists(path):
                    studio.load_vae(path)

        elif choice == "3":
            # Załaduj LoRA
            path = input("\nŚcieżka do LoRA .safetensors: ").strip()
            if os.path.exists(path):
                weight = input("Waga (0.1-1.5, domyślnie 1.0): ").strip()
                weight = float(weight) if weight else 1.0
                studio.load_lora(path, weight=weight)
            else:
                print("❌ Plik nie istnieje!")

        elif choice == "4":
            # Pokaż modele
            studio.list_loaded_models()

        elif choice == "5":
            # Generuj
            if not studio.pipe:
                print("❌ Najpierw załaduj checkpoint (opcja 1)!")
                continue

            prompt = input("\nPrompt (opis obrazu): ").strip()
            negative = input("Negative prompt (Enter = pomiń): ").strip()

            width = input("Szerokość (Enter = 512): ").strip()
            width = int(width) if width else 512

            height = input("Wysokość (Enter = 512): ").strip()
            height = int(height) if height else 512

            output = input("Nazwa pliku (Enter = output.png): ").strip()
            output = output if output else "output.png"

            studio.generate(
                prompt=prompt,
                negative_prompt=negative,
                width=width,
                height=height,
                output_path=output
            )

        elif choice == "6":
            # Dodaj tekst
            img_path = input("\nŚcieżka do obrazu: ").strip()
            if not os.path.exists(img_path):
                print("❌ Plik nie istnieje!")
                continue

            text = input("Tekst (POLSKIE ZNAKI OK!): ").strip()

            x = input("Pozycja X (Enter = 50): ").strip()
            x = int(x) if x else 50

            y = input("Pozycja Y (Enter = 50): ").strip()
            y = int(y) if y else 50

            size = input("Rozmiar czcionki (Enter = 48): ").strip()
            size = int(size) if size else 48

            format_out = input("Format (png/jpg/webp/tiff, Enter = png): ").strip()
            format_out = format_out if format_out else "png"

            studio.add_text_to_image(
                image_path=img_path,
                text=text,
                position=(x, y),
                font_size=size,
                output_format=format_out
            )

        elif choice == "7":
            # Watermark
            img_path = input("\nŚcieżka do obrazu: ").strip()
            if not os.path.exists(img_path):
                print("❌ Plik nie istnieje!")
                continue

            text = input("Tekst watermarku: ").strip()

            print("\nPozycja:")
            print("  1) Prawy dolny róg")
            print("  2) Lewy dolny róg")
            print("  3) Prawy górny róg")
            print("  4) Lewy górny róg")
            print("  5) Środek")

            pos_choice = input("Wybierz (1-5): ").strip()
            positions = {
                '1': 'bottom-right',
                '2': 'bottom-left',
                '3': 'top-right',
                '4': 'top-left',
                '5': 'center'
            }
            position = positions.get(pos_choice, 'bottom-right')

            studio.add_watermark(
                image_path=img_path,
                watermark_text=text,
                position=position
            )

        elif choice == "8":
            # Konwertuj format
            img_path = input("\nŚcieżka do obrazu: ").strip()
            if not os.path.exists(img_path):
                print("❌ Plik nie istnieje!")
                continue

            print("\nFormat docelowy:")
            print("  1) PNG (bezstratny)")
            print("  2) JPG (mniejszy rozmiar)")
            print("  3) WEBP (najlepsze kompresja)")
            print("  4) TIFF (archiwalna jakość)")

            fmt_choice = input("Wybierz (1-4): ").strip()
            formats = {'1': 'png', '2': 'jpg', '3': 'webp', '4': 'tiff'}
            fmt = formats.get(fmt_choice, 'png')

            quality = 95
            if fmt in ['jpg', 'webp']:
                q = input("Jakość (1-100, Enter = 95): ").strip()
                quality = int(q) if q else 95

            studio.convert_format(img_path, fmt, quality=quality)

        elif choice == "9":
            # Batch convert
            folder = input("\nŚcieżka do folderu: ").strip()
            if not os.path.exists(folder):
                print("❌ Folder nie istnieje!")
                continue

            print("\nFormat docelowy:")
            print("  1) PNG  2) JPG  3) WEBP  4) TIFF")

            fmt_choice = input("Wybierz (1-4): ").strip()
            formats = {'1': 'png', '2': 'jpg', '3': 'webp', '4': 'tiff'}
            fmt = formats.get(fmt_choice, 'png')

            studio.batch_convert(folder, fmt)

    print("\n✅ Program zakończony!")


# ============================================================================
#                           URUCHOM
# ============================================================================

if __name__ == "__main__":
    interactive_menu()
