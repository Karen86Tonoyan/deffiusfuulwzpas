# 🎨 ALPHA IMAGE STUDIO

**Profesjonalny pakiet AI do generowania i edycji obrazów - wszystko w jednym!**

![Version](https://img.shields.io/badge/version-1.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## ✨ Funkcje

### 🎨 Moduły

| Moduł | Funkcja | Plik |
|-------|---------|------|
| **Generator** | Generowanie obrazów z tekstu (SDXL, SD 2.1, SD 1.5) | `ai_generator.py` |
| **Clothes** | Wirtualna przymierzalnia - zmiana ubrań | `ai_clothes.py` |
| **Text PL** | Dodawanie polskiego tekstu i watermarków | `ai_text_pl.py` |
| **Upscale** | Zwiększanie rozdzielczości do 4K/8K | `ai_upscale.py` |
| **GUI** | Profesjonalny interfejs Gradio | `alpha_studio_ui.py` |

### 🚀 Możliwości

✅ **Generowanie obrazów** - SDXL, SD 2.1, SD 1.5
✅ **Zmiana ubrań** - wirtualna przymierzalnia AI
✅ **Polski tekst** - pełne wsparcie dla ąćęłńóśźż
✅ **Upscaling** - do 4K (3840x2160) i 8K (7680x4320)
✅ **Watermarki** - profesjonalne znaki wodne
✅ **Enhance** - szybka poprawa jakości
✅ **GUI** - piękny interfejs w przeglądarce
✅ **Batch** - przetwarzanie wielu plików

---

## 📦 Instalacja

### Krok 1: Wymagania

- Python 3.8+
- 10GB wolnego miejsca (modele AI)
- GPU NVIDIA (zalecane, nie wymagane)

### Krok 2: Automatyczna instalacja

```bash
INSTALL_ALL.bat
```

**LUB ręcznie:**

```bash
# PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# AI i obrazy
pip install diffusers transformers accelerate safetensors pillow

# GUI
pip install gradio
```

---

## 🚀 Szybki Start

### Opcja 1: GUI (Najłatwiejsze!)

```bash
python alpha_studio_ui.py
```

Otwórz przeglądarkę: `http://localhost:7860`

### Opcja 2: Używaj modułów bezpośrednio

```python
# Generuj obraz
from ai_generator import generate_image

generate_image(
    prompt="beautiful landscape, mountains, sunset, 4k",
    res="1024x1024",
    model="sdxl"
)

# Zmień ubrania
from ai_clothes import change_clothes

change_clothes(
    image_path="selfie.jpg",
    mask_path="mask.png",
    prompt="red t-shirt, casual, high quality"
)

# Dodaj polski tekst
from ai_text_pl import add_polish_text

add_polish_text(
    image_path="photo.jpg",
    text="POLSKIE ZNAKI: ąćęłńóśźż",
    position=(50, 50),
    font_size=64
)

# Upscale do 4K
from ai_upscale import upscale_to_4k

upscale_to_4k("photo.jpg", output="photo_4k.png")
```

---

## 📖 Dokumentacja Modułów

### 1️⃣ ai_generator.py - Generowanie Obrazów

```python
generate_image(
    prompt="beautiful cat, digital art, detailed",
    negative_prompt="ugly, blurry, low quality",
    res="1024x1024",    # 512x512, 768x768, 1024x1024
    model="sdxl",       # sdxl, sd21, sd15
    output="cat.png"
)
```

**Modele:**
- `sdxl` - Najlepsza jakość (wolniejszy)
- `sd21` - Dobra jakość (średni)
- `sd15` - Szybki (podstawowy)

**Tips:**
- Używaj angielskich słów kluczowych
- Dodaj: `"4k", "detailed", "masterpiece"`
- Określ styl: `"oil painting", "digital art", "photorealistic"`

---

### 2️⃣ ai_clothes.py - Wirtualna Przymierzalnia

```python
change_clothes(
    image_path="selfie.jpg",
    mask_path="mask.png",
    prompt="elegant black suit, formal, professional",
    guidance=7.5,      # 5-15 (wyżej = bardziej zgodny z promptem)
    steps=35,          # 25-50 (wyżej = lepsza jakość)
    output="result.png"
)
```

**Jak stworzyć maskę:**
1. Otwórz zdjęcie w Paint
2. Zamaluj NA BIAŁO tylko ubranie
3. Reszta niech będzie CZARNA
4. Zapisz jako PNG

**Przykładowe prompty:**
- `"red t-shirt, casual, cotton"`
- `"elegant dress, blue, evening gown"`
- `"business suit, formal, professional"`

---

### 3️⃣ ai_text_pl.py - Polski Tekst i Watermarki

**Dodaj tekst:**

```python
add_polish_text(
    image_path="photo.jpg",
    text="POLSKIE ZNAKI: ąćęłńóśźż",
    position=(50, 50),       # (x, y)
    font_size=64,
    color=(255, 255, 255),   # RGB white
    shadow=True,             # Cień
    output="with_text.png"
)
```

**Dodaj watermark:**

```python
add_watermark(
    image_path="photo.jpg",
    watermark_text="© 2025 Moja Firma",
    position='bottom-right',  # top-left, top-right, bottom-left, bottom-right, center
    font_size=24,
    opacity=0.5,             # 0.0-1.0
    output="watermarked.png"
)
```

---

### 4️⃣ ai_upscale.py - Upscaling i Enhance

**Upscale 4x:**

```python
upscale_4x(
    image_path="photo.jpg",
    method="ai",        # "ai" (wolniejsze, lepsze) lub "fast" (szybkie)
    output="photo_4x.png"
)
```

**Upscale do 4K:**

```python
upscale_to_4k(
    image_path="photo.jpg",
    output="photo_4k.png"
)
# Wynik: 3840x2160
```

**Szybka poprawa jakości:**

```python
enhance_photo(
    image_path="selfie.jpg",
    sharpness=1.3,    # 0.5-2.0
    color=1.2,        # Saturacja
    contrast=1.1,
    brightness=1.0,
    output="enhanced.jpg"
)
```

---

## 🎨 GUI - Interfejs Graficzny

### Uruchomienie

```bash
python alpha_studio_ui.py
```

### Zakładki

1. **✨ Generuj Obraz** - Text2Image
2. **👔 Zmień Ubranie** - Wirtualna przymierzalnia
3. **✍️ Polski Tekst** - Dodaj tekst z ąćęłńóśźż
4. **💧 Watermark** - Dodaj znak wodny
5. **📈 Upscaling** - Zwiększ rozdzielczość
6. **✨ Enhance** - Popraw jakość

### Funkcje GUI

- ✅ Drag & Drop obrazów
- ✅ Live preview
- ✅ Slidery do kontroli parametrów
- ✅ Zapisywanie wyników
- ✅ Presety i szybkie akcje
- ✅ Progress bar
- ✅ Dark mode

---

## 📁 Struktura Projektu

```
diffusers/
├── alpha_studio_ui.py      ← GUI (uruchom to!)
├── ai_generator.py          ← Text2Image
├── ai_clothes.py            ← Zmiana ubrań
├── ai_text_pl.py            ← Polski tekst
├── ai_upscale.py            ← Upscaling
│
├── INSTALL_ALL.bat          ← Instalator
├── README_ALPHA_STUDIO.md   ← Ten plik
│
├── wirtualna_przymierzalnia.py  ← Standalone clothes changer
├── ultimate_ai_tools.py         ← Wszystko w jednym (terminal)
├── ai_studio_ultimate.py        ← Ultimate z checkpointami
│
└── generate_image.py        ← Prosty generator (dla początkujących)
```

---

## 💡 Przykłady Użycia

### Przykład 1: Wygeneruj i dodaj tekst

```python
from ai_generator import generate_image
from ai_text_pl import add_polish_text

# 1. Wygeneruj tło
generate_image(
    prompt="abstract background, colorful, modern",
    res="1024x1024",
    model="sdxl",
    output="background.png"
)

# 2. Dodaj polski tekst
add_polish_text(
    image_path="background.png",
    text="MOJA FIRMA\nNajlepsza w Polsce!",
    position=(100, 100),
    font_size=72,
    output="final.png"
)
```

### Przykład 2: Selfie → Makijaż → 4K

```python
from ai_clothes import change_clothes
from ai_upscale import upscale_to_4k

# 1. Dodaj makijaż
change_clothes(
    image_path="selfie.jpg",
    mask_path="face_mask.png",
    prompt="professional makeup, glamour, beauty",
    output="with_makeup.png"
)

# 2. Upscale do 4K
upscale_to_4k("with_makeup.png", "final_4k.png")
```

### Przykład 3: Batch Watermark

```python
from ai_text_pl import add_watermark
from pathlib import Path

# Dodaj watermark do wszystkich zdjęć
for img in Path("photos").glob("*.jpg"):
    add_watermark(
        str(img),
        "© 2025 Moja Firma",
        position='bottom-right',
        output=f"watermarked/{img.name}"
    )
```

---

## 🔧 Rozwiązywanie Problemów

### Błąd: "CUDA out of memory"

**Rozwiązanie:**
- Zmniejsz rozdzielczość (512x512 zamiast 1024x1024)
- Zamknij inne programy
- Użyj modelu `sd15` zamiast `sdxl`

### Błąd: "No module named 'diffusers'"

**Rozwiązanie:**
```bash
pip install diffusers transformers torch
```

### GUI nie uruchamia się

**Rozwiązanie:**
```bash
pip install gradio
python alpha_studio_ui.py
```

### Polskie znaki nie działają

**Rozwiązanie:**
- Zainstaluj czcionkę Arial (powinna być w Windows)
- Lub podaj własną czcionkę TTF:
```python
add_polish_text(..., font_path="moja_czcionka.ttf")
```

### Upscaling bardzo wolny

**Rozwiązanie:**
- Użyj `method="fast"` zamiast `"ai"`
- Lub użyj `enhance_photo()` zamiast upscalingu

---

## ⚙️ Konfiguracja

### Zmiana domyślnego modelu

W `ai_generator.py`:
```python
# Zmień:
model="sdxl"
# Na:
model="sd21"  # Szybszy
```

### Własne checkpointy

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_single_file(
    "models/my_model.safetensors"
)
```

### GPU vs CPU

Automatycznie wykrywa GPU. Wymusz CPU:
```python
device = "cpu"  # Zamiast "cuda"
```

---

## 📊 Wymagania Systemowe

| Funkcja | Minimalne | Zalecane |
|---------|-----------|----------|
| **Generowanie** | 8GB RAM, CPU | 16GB RAM, GPU 6GB+ |
| **Upscaling AI** | 12GB RAM, GPU 4GB | 16GB RAM, GPU 8GB+ |
| **Upscaling Fast** | 4GB RAM | 8GB RAM |
| **Enhance** | 2GB RAM | 4GB RAM |
| **Text/Watermark** | 2GB RAM | 4GB RAM |

**Czas generowania (przykładowo):**

| Operacja | CPU | GPU (RTX 3060) |
|----------|-----|----------------|
| Text2Img 1024x1024 | ~3min | ~15sek |
| Clothes Change | ~4min | ~25sek |
| Upscale 4x (AI) | ~5min | ~30sek |
| Enhance | <1sek | <1sek |

---

## 🎯 Roadmap

- [ ] Video generation (AnimateDiff)
- [ ] ControlNet integration
- [ ] Multi-language UI
- [ ] Cloud deployment
- [ ] API endpoints
- [ ] Mobile app

---

## 📚 Zasoby

**Inspiracje:**
- [Lexica.art](https://lexica.art) - Galeria promptów
- [PromptHero](https://prompthero.com) - Baza promptów
- [Civitai](https://civitai.com) - Modele i przykłady

**Dokumentacja:**
- [Diffusers Docs](https://huggingface.co/docs/diffusers)
- [Gradio Docs](https://gradio.app/docs)

**Community:**
- [r/StableDiffusion](https://reddit.com/r/StableDiffusion)
- [Hugging Face Discord](https://discord.gg/huggingface)

---

## ⚖️ Licencja

MIT License - wolne użycie komercyjne i niekomercyjne

**Modele AI:**
- Stable Diffusion: CreativeML Open RAIL++-M
- Modele z HuggingFace: zgodnie z ich licencjami

---

## 🙏 Podziękowania

- Stability AI - Stable Diffusion
- Hugging Face - Diffusers
- Gradio - GUI framework

---

## 📧 Kontakt

Problemy? Pytania?
- GitHub Issues
- Discord: (link)

---

**Miłego tworzenia! 🎨✨**

Made with ❤️ in Poland 🇵🇱
