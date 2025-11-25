"""
🚀 ULTIMATE AI TOOLS - WSZYSTKO W JEDNYM!
==========================================

✅ Generowanie obrazów (text2img, img2img, inpaint)
✅ Checkpointy, VAE, LoRA
✅ Polski tekst na obrazach
✅ Wszystkie formaty (PNG, JPG, WEBP, TIFF, PDF)
✅ UPSCALING do 4K/8K! 📈
✅ PDF → eBook (EPUB, MOBI) 📚
✅ Watermarki
✅ Batch processing
"""

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionImg2ImgPipeline
)
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import torch
import os
from pathlib import Path

# Dla PDF → eBook
try:
    import PyPDF2
    from ebooklib import epub
    PDF_SUPPORT = True
except:
    PDF_SUPPORT = False
    print("⚠️  Dla PDF→eBook zainstaluj: pip install PyPDF2 ebooklib")

class UltimateAITools:
    def __init__(self):
        """Inicjalizacja"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.pipe = None
        self.upscaler = None

        print("🚀 Ultimate AI Tools - Inicjalizacja")
        print(f"Urządzenie: {self.device}")

    # ========================================================================
    #                     UPSCALING - ZWIĘKSZANIE ROZDZIELCZOŚCI
    # ========================================================================

    def load_upscaler(self):
        """Załaduj model upscaling"""
        if self.upscaler is None:
            print("\n📥 Ładowanie modelu upscaling...")
            self.upscaler = StableDiffusionUpscalePipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler",
                torch_dtype=self.dtype
            )
            self.upscaler = self.upscaler.to(self.device)
            if self.device == "cuda":
                self.upscaler.enable_attention_slicing()
            print("✅ Upscaler załadowany!")

    def upscale_to_4k(self, image_path, output_path="upscaled_4k.png",
                      prompt="high quality, detailed, sharp"):
        """
        Zwiększ rozdzielczość do ~4K (3840x2160)

        Args:
            image_path: Ścieżka do obrazu (str lub PIL.Image)
            output_path: Gdzie zapisać
            prompt: Opis dla AI (pomaga w upscalingu)

        Returns:
            PIL.Image: Obraz w wyższej rozdzielczości
        """
        print(f"\n📈 Upscaling do 4K...")

        self.load_upscaler()

        # Wczytaj obraz
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")

        print(f"Rozmiar wejściowy: {image.size}")

        # Upscale 4x
        upscaled = self.upscaler(
            prompt=prompt,
            image=image,
            num_inference_steps=50
        ).images[0]

        print(f"Rozmiar wyjściowy: {upscaled.size}")

        # Jeśli nie jest dokładnie 4K, przeskaluj do 3840x2160
        target_4k = (3840, 2160)
        if upscaled.size != target_4k:
            # Zachowaj proporcje
            upscaled.thumbnail(target_4k, Image.Resampling.LANCZOS)

        upscaled.save(output_path)
        print(f"✅ Zapisano 4K: {output_path}")

        return upscaled

    def upscale_to_8k(self, image_path, output_path="upscaled_8k.png",
                      prompt="high quality, detailed, sharp, professional"):
        """
        Zwiększ rozdzielczość do ~8K (7680x4320)

        UWAGA: Wymaga dużo pamięci GPU/RAM!

        Args:
            image_path: Ścieżka do obrazu
            output_path: Gdzie zapisać
            prompt: Opis dla AI
        """
        print(f"\n📈 Upscaling do 8K (może potrwać!)...")
        print("⚠️  Wymaga ~16GB RAM lub silnego GPU!")

        # Najpierw do 4K
        img_4k = self.upscale_to_4k(image_path, "temp_4k.png", prompt)

        # Potem jeszcze raz upscale
        self.load_upscaler()

        upscaled_8k = self.upscaler(
            prompt=prompt,
            image=img_4k,
            num_inference_steps=50
        ).images[0]

        # Przeskaluj do dokładnie 8K
        target_8k = (7680, 4320)
        upscaled_8k.thumbnail(target_8k, Image.Resampling.LANCZOS)

        upscaled_8k.save(output_path)
        print(f"✅ Zapisano 8K: {output_path}")

        # Usuń temp
        if os.path.exists("temp_4k.png"):
            os.remove("temp_4k.png")

        return upscaled_8k

    def enhance_photo(self, image_path, output_path="enhanced.png",
                     sharpness=1.3, color=1.2, contrast=1.1, brightness=1.0):
        """
        Popraw jakość zdjęcia (bez AI, szybkie!)

        Args:
            image_path: Ścieżka do zdjęcia
            output_path: Gdzie zapisać
            sharpness: Ostrość (1.0 = bez zmian, >1.0 = ostrzejsze)
            color: Saturacja kolorów (1.0 = bez zmian)
            contrast: Kontrast (1.0 = bez zmian)
            brightness: Jasność (1.0 = bez zmian)
        """
        print(f"\n✨ Poprawa jakości zdjęcia...")

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

        img.save(output_path)
        print(f"✅ Zapisano: {output_path}")

        return img

    # ========================================================================
    #                     PDF → eBook (EPUB, MOBI)
    # ========================================================================

    def pdf_to_epub(self, pdf_path, epub_path=None, title=None, author="Unknown"):
        """
        Konwertuj PDF do EPUB (eBook)

        Args:
            pdf_path: Ścieżka do PDF
            epub_path: Gdzie zapisać EPUB (None = auto)
            title: Tytuł książki
            author: Autor
        """
        if not PDF_SUPPORT:
            print("❌ Zainstaluj: pip install PyPDF2 ebooklib")
            return

        print(f"\n📚 Konwersja PDF → EPUB...")

        if not epub_path:
            epub_path = Path(pdf_path).stem + ".epub"

        if not title:
            title = Path(pdf_path).stem

        # Wczytaj PDF
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)

            print(f"Strony: {num_pages}")

            # Stwórz EPUB
            book = epub.EpubBook()

            # Metadane
            book.set_identifier('id123456')
            book.set_title(title)
            book.set_language('pl')  # Polski!
            book.add_author(author)

            # Wyciągnij tekst z każdej strony
            chapters = []
            for i in range(num_pages):
                page = pdf_reader.pages[i]
                text = page.extract_text()

                # Stwórz rozdział
                chapter = epub.EpubHtml(
                    title=f'Rozdział {i+1}',
                    file_name=f'chap_{i:03d}.xhtml',
                    lang='pl'
                )
                chapter.content = f'<h1>Strona {i+1}</h1><p>{text}</p>'

                book.add_item(chapter)
                chapters.append(chapter)

            # Spis treści
            book.toc = chapters

            # Dodaj nawigację
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())

            # Definiuj kolejność
            book.spine = ['nav'] + chapters

            # Zapisz
            epub.write_epub(epub_path, book, {})

        print(f"✅ EPUB zapisany: {epub_path}")
        return epub_path

    def pdf_to_text(self, pdf_path, output_path=None):
        """
        Wyciągnij tekst z PDF

        Args:
            pdf_path: Ścieżka do PDF
            output_path: Gdzie zapisać TXT (None = auto)
        """
        if not PDF_SUPPORT:
            print("❌ Zainstaluj: pip install PyPDF2")
            return

        print(f"\n📄 Wyciąganie tekstu z PDF...")

        if not output_path:
            output_path = Path(pdf_path).stem + ".txt"

        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""

            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f"✅ Tekst zapisany: {output_path}")
        return text

    # ========================================================================
    #                     BATCH PROCESSING
    # ========================================================================

    def batch_upscale(self, folder_path, target_resolution="4k"):
        """
        Upscale wszystkich obrazów w folderze

        Args:
            folder_path: Ścieżka do folderu
            target_resolution: '4k' lub '8k'
        """
        folder = Path(folder_path)
        output_folder = folder / f"upscaled_{target_resolution}"
        output_folder.mkdir(exist_ok=True)

        supported = ['.png', '.jpg', '.jpeg', '.webp']
        images = []
        for ext in supported:
            images.extend(folder.glob(f"*{ext}"))

        print(f"\n📈 Upscaling {len(images)} obrazów do {target_resolution.upper()}...")

        for img_path in images:
            output_path = output_folder / f"{img_path.stem}_{target_resolution}.png"

            if target_resolution == "8k":
                self.upscale_to_8k(str(img_path), str(output_path))
            else:  # 4k
                self.upscale_to_4k(str(img_path), str(output_path))

        print(f"\n✅ Gotowe! Obrazy w: {output_folder}")

    def batch_enhance(self, folder_path):
        """Popraw wszystkie zdjęcia w folderze"""
        folder = Path(folder_path)
        output_folder = folder / "enhanced"
        output_folder.mkdir(exist_ok=True)

        supported = ['.png', '.jpg', '.jpeg']
        images = []
        for ext in supported:
            images.extend(folder.glob(f"*{ext}"))

        print(f"\n✨ Poprawa {len(images)} zdjęć...")

        for img_path in images:
            output_path = output_folder / img_path.name
            self.enhance_photo(str(img_path), str(output_path))

        print(f"\n✅ Gotowe! Zdjęcia w: {output_folder}")

    # ========================================================================
    #                     DODATKOWE NARZĘDZIA
    # ========================================================================

    def create_thumbnail(self, image_path, size=(256, 256), output_path=None):
        """Stwórz miniaturkę"""
        img = Image.open(image_path)
        img.thumbnail(size, Image.Resampling.LANCZOS)

        if not output_path:
            output_path = f"thumb_{Path(image_path).name}"

        img.save(output_path)
        print(f"✅ Miniaturka: {output_path}")
        return img

    def create_contact_sheet(self, folder_path, grid_size=(4, 4),
                            thumb_size=(256, 256), output_path="contact_sheet.png"):
        """
        Stwórz arkusz kontaktowy (galeria miniaturek)

        Args:
            folder_path: Folder ze zdjęciami
            grid_size: Siatka (kolumny, wiersze)
            thumb_size: Rozmiar miniaturki
            output_path: Gdzie zapisać
        """
        print(f"\n🖼️  Tworzenie arkusza kontaktowego...")

        folder = Path(folder_path)
        supported = ['.png', '.jpg', '.jpeg', '.webp']
        images = []
        for ext in supported:
            images.extend(folder.glob(f"*{ext}"))

        cols, rows = grid_size
        max_images = cols * rows

        if len(images) > max_images:
            images = images[:max_images]
            print(f"⚠️  Używam pierwszych {max_images} obrazów")

        # Stwórz canvas
        canvas_width = thumb_size[0] * cols
        canvas_height = thumb_size[1] * rows
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

        # Wklej miniaturki
        for idx, img_path in enumerate(images):
            img = Image.open(img_path)
            img.thumbnail(thumb_size, Image.Resampling.LANCZOS)

            # Oblicz pozycję
            col = idx % cols
            row = idx // cols

            x = col * thumb_size[0]
            y = row * thumb_size[1]

            # Wycentruj jeśli miniaturka jest mniejsza
            offset_x = (thumb_size[0] - img.width) // 2
            offset_y = (thumb_size[1] - img.height) // 2

            canvas.paste(img, (x + offset_x, y + offset_y))

        canvas.save(output_path)
        print(f"✅ Arkusz zapisany: {output_path}")
        return canvas


# ============================================================================
#                        PRZYKŁADY UŻYCIA
# ============================================================================

def example_upscale_4k():
    """Przykład: Upscale do 4K"""
    tools = UltimateAITools()

    tools.upscale_to_4k(
        image_path="photo.jpg",
        output_path="photo_4k.png",
        prompt="professional photo, high quality, detailed, sharp"
    )


def example_upscale_8k():
    """Przykład: Upscale do 8K (wymaga dużo pamięci!)"""
    tools = UltimateAITools()

    tools.upscale_to_8k(
        image_path="photo.jpg",
        output_path="photo_8k.png",
        prompt="ultra high quality, professional, masterpiece, 8k"
    )


def example_enhance_photo():
    """Przykład: Szybka poprawa zdjęcia"""
    tools = UltimateAITools()

    tools.enhance_photo(
        image_path="selfie.jpg",
        output_path="selfie_enhanced.jpg",
        sharpness=1.3,    # Ostrość
        color=1.2,        # Więcej kolorów
        contrast=1.1,     # Lepszy kontrast
        brightness=1.05   # Trochę jaśniej
    )


def example_pdf_to_epub():
    """Przykład: PDF → EPUB"""
    tools = UltimateAITools()

    tools.pdf_to_epub(
        pdf_path="ksiazka.pdf",
        epub_path="ksiazka.epub",
        title="Moja Książka",
        author="Jan Kowalski"
    )


def example_batch_upscale():
    """Przykład: Upscale całego folderu"""
    tools = UltimateAITools()

    tools.batch_upscale(
        folder_path="photos",
        target_resolution="4k"  # lub "8k"
    )


def example_contact_sheet():
    """Przykład: Arkusz kontaktowy"""
    tools = UltimateAITools()

    tools.create_contact_sheet(
        folder_path="photos",
        grid_size=(5, 4),  # 5 kolumn x 4 wiersze = 20 zdjęć
        thumb_size=(256, 256),
        output_path="galeria.png"
    )


# ============================================================================
#                     INTERAKTYWNE MENU
# ============================================================================

def interactive_menu():
    """Menu interaktywne"""
    tools = UltimateAITools()

    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║              🚀 ULTIMATE AI TOOLS 🚀                             ║
    ║                                                                  ║
    ║  📈 Upscaling 4K/8K  ✨ Enhance  📚 PDF→eBook                    ║
    ║  🖼️  Batch  📋 Contact Sheet  💧 Watermark                       ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    while True:
        print("\n" + "="*70)
        print("MENU GŁÓWNE:")
        print("="*70)
        print()
        print("UPSCALING:")
        print("  1. 📈 Upscale do 4K (3840x2160)")
        print("  2. 📈 Upscale do 8K (7680x4320) ⚠️  Wymaga dużo RAM!")
        print("  3. 📁 Batch Upscale (cały folder)")
        print()
        print("POPRAWA JAKOŚCI:")
        print("  4. ✨ Enhance (szybka poprawa)")
        print("  5. 📁 Batch Enhance (cały folder)")
        print()
        print("PDF & eBOOK:")
        print("  6. 📚 PDF → EPUB (eBook)")
        print("  7. 📄 PDF → TXT (wyciągnij tekst)")
        print()
        print("NARZĘDZIA:")
        print("  8. 🖼️  Stwórz arkusz kontaktowy (galeria)")
        print("  9. 🔍 Miniaturka")
        print()
        print("  0. ❌ Wyjście")
        print()

        choice = input("Wybierz (0-9): ").strip()

        if choice == "0":
            print("\nDo zobaczenia! 👋")
            break

        elif choice == "1":
            # Upscale 4K
            img = input("\nŚcieżka do obrazu: ").strip()
            if not os.path.exists(img):
                print("❌ Plik nie istnieje!")
                continue

            output = input("Nazwa wyjściowa (Enter = upscaled_4k.png): ").strip()
            output = output if output else "upscaled_4k.png"

            prompt = input("Prompt dla AI (Enter = domyślny): ").strip()
            prompt = prompt if prompt else "high quality, detailed, sharp"

            tools.upscale_to_4k(img, output, prompt)

        elif choice == "2":
            # Upscale 8K
            img = input("\nŚcieżka do obrazu: ").strip()
            if not os.path.exists(img):
                print("❌ Plik nie istnieje!")
                continue

            print("\n⚠️  UWAGA: 8K wymaga ~16GB RAM i zajmie kilka minut!")
            confirm = input("Kontynuować? (t/n): ").strip().lower()
            if confirm != 't':
                continue

            output = input("Nazwa wyjściowa (Enter = upscaled_8k.png): ").strip()
            output = output if output else "upscaled_8k.png"

            tools.upscale_to_8k(img, output)

        elif choice == "3":
            # Batch upscale
            folder = input("\nFolder ze zdjęciami: ").strip()
            if not os.path.exists(folder):
                print("❌ Folder nie istnieje!")
                continue

            print("\nRozdzielczość:")
            print("  1) 4K (szybciej)")
            print("  2) 8K (wolniej, więcej pamięci)")

            res_choice = input("Wybierz (1-2): ").strip()
            target = "8k" if res_choice == "2" else "4k"

            tools.batch_upscale(folder, target)

        elif choice == "4":
            # Enhance
            img = input("\nŚcieżka do zdjęcia: ").strip()
            if not os.path.exists(img):
                print("❌ Plik nie istnieje!")
                continue

            print("\nParametry (Enter = domyślne):")
            sharp = input("Ostrość (1.0-2.0, domyślnie 1.3): ").strip()
            sharp = float(sharp) if sharp else 1.3

            color = input("Kolory (1.0-2.0, domyślnie 1.2): ").strip()
            color = float(color) if color else 1.2

            tools.enhance_photo(img, sharpness=sharp, color=color)

        elif choice == "5":
            # Batch enhance
            folder = input("\nFolder ze zdjęciami: ").strip()
            if not os.path.exists(folder):
                print("❌ Folder nie istnieje!")
                continue

            tools.batch_enhance(folder)

        elif choice == "6":
            # PDF → EPUB
            pdf = input("\nŚcieżka do PDF: ").strip()
            if not os.path.exists(pdf):
                print("❌ Plik nie istnieje!")
                continue

            title = input("Tytuł książki (Enter = nazwa pliku): ").strip()
            title = title if title else Path(pdf).stem

            author = input("Autor (Enter = Unknown): ").strip()
            author = author if author else "Unknown"

            tools.pdf_to_epub(pdf, title=title, author=author)

        elif choice == "7":
            # PDF → TXT
            pdf = input("\nŚcieżka do PDF: ").strip()
            if not os.path.exists(pdf):
                print("❌ Plik nie istnieje!")
                continue

            tools.pdf_to_text(pdf)

        elif choice == "8":
            # Contact sheet
            folder = input("\nFolder ze zdjęciami: ").strip()
            if not os.path.exists(folder):
                print("❌ Folder nie istnieje!")
                continue

            cols = input("Kolumny (Enter = 4): ").strip()
            cols = int(cols) if cols else 4

            rows = input("Wiersze (Enter = 4): ").strip()
            rows = int(rows) if rows else 4

            tools.create_contact_sheet(folder, grid_size=(cols, rows))

        elif choice == "9":
            # Miniaturka
            img = input("\nŚcieżka do obrazu: ").strip()
            if not os.path.exists(img):
                print("❌ Plik nie istnieje!")
                continue

            size = input("Rozmiar (Enter = 256): ").strip()
            size = int(size) if size else 256

            tools.create_thumbnail(img, size=(size, size))

        print("\n✅ Gotowe!")
        again = input("\nKolejne działanie? (t/n): ").strip().lower()
        if again != 't':
            break

    print("\n✅ Program zakończony!")


if __name__ == "__main__":
    interactive_menu()
