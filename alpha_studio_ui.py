"""
🎨 ALPHA IMAGE STUDIO - Profesjonalne GUI w Gradio
====================================================

Wszystkie funkcje w jednym miejscu:
- Generowanie obrazów (SDXL)
- Zmiana ubrań
- Polski tekst
- Upscaling 4K
- Watermarki
- I więcej!
"""

import gradio as gr
from ai_generator import generate_image
from ai_clothes import change_clothes
from ai_text_pl import add_polish_text, add_watermark
from ai_upscale import upscale_4x, enhance_photo, upscale_to_4k
from PIL import Image
import os

# Globalne style
CSS = """
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
h1 {
    text-align: center;
    color: #FFD700;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    font-size: 3em;
    margin: 20px 0;
}
.tab-nav button {
    font-size: 16px;
    font-weight: bold;
}
"""

def create_ui():
    """Stwórz interfejs Gradio"""

    with gr.Blocks(css=CSS, title="🎨 ALPHA IMAGE STUDIO", theme=gr.themes.Soft()) as app:

        gr.Markdown("""
        # 🎨 ALPHA IMAGE STUDIO
        ### Profesjonalna edycja i generowanie obrazów AI
        ---
        """)

        with gr.Tabs():

            # ================================================================
            # TAB 1: GENEROWANIE OBRAZÓW
            # ================================================================
            with gr.Tab("✨ Generuj Obraz"):
                gr.Markdown("### Stwórz obraz z opisu tekstowego")

                with gr.Row():
                    with gr.Column():
                        gen_prompt = gr.Textbox(
                            label="📝 Prompt (opis obrazu)",
                            placeholder="beautiful landscape, mountains, sunset, photorealistic, 4k...",
                            lines=5
                        )
                        gen_negative = gr.Textbox(
                            label="❌ Negative Prompt (czego unikać)",
                            placeholder="ugly, blurry, low quality, bad anatomy...",
                            lines=3
                        )

                        with gr.Row():
                            gen_model = gr.Dropdown(
                                choices=["sdxl", "sd21", "sd15"],
                                value="sdxl",
                                label="🎨 Model"
                            )
                            gen_res = gr.Dropdown(
                                choices=["512x512", "768x768", "1024x1024", "1024x768", "768x1024"],
                                value="1024x1024",
                                label="📐 Rozdzielczość"
                            )

                        gen_btn = gr.Button("🚀 GENERUJ OBRAZ", variant="primary", size="lg")

                    with gr.Column():
                        gen_output = gr.Image(label="Wygenerowany obraz", type="filepath")

                gen_btn.click(
                    fn=generate_image,
                    inputs=[gen_prompt, gen_negative, gen_res, gen_model],
                    outputs=gen_output
                )

                gr.Markdown("""
                **💡 Wskazówki:**
                - Używaj angielskich słów: "beautiful", "detailed", "4k", "masterpiece"
                - Dodaj styl: "oil painting", "digital art", "photorealistic"
                - Model SDXL = najlepsza jakość (wolniejszy)
                """)

            # ================================================================
            # TAB 2: ZMIANA UBRAŃ
            # ================================================================
            with gr.Tab("👔 Zmień Ubranie"):
                gr.Markdown("### Wirtualna przymierzalnia - zmień ubrania na zdjęciu")

                with gr.Row():
                    with gr.Column():
                        cloth_img = gr.Image(label="📸 Zdjęcie osoby", type="filepath")
                        cloth_mask = gr.Image(label="🎭 Maska (białe = zmień)", type="filepath")
                        cloth_prompt = gr.Textbox(
                            label="👕 Opis nowych ubrań",
                            placeholder="red t-shirt, casual, high quality...",
                            lines=3
                        )

                        with gr.Row():
                            cloth_guidance = gr.Slider(
                                minimum=5, maximum=15, value=7.5, step=0.5,
                                label="⚙️ Guidance (zgodność z promptem)"
                            )
                            cloth_steps = gr.Slider(
                                minimum=25, maximum=50, value=35, step=5,
                                label="🔄 Kroki (jakość)"
                            )

                        cloth_btn = gr.Button("👔 ZMIEŃ UBRANIE", variant="primary", size="lg")

                    with gr.Column():
                        cloth_output = gr.Image(label="Wynik", type="filepath")

                cloth_btn.click(
                    fn=lambda img, mask, prompt, guid, steps: change_clothes(
                        img, mask, prompt, guidance=guid, steps=steps
                    ),
                    inputs=[cloth_img, cloth_mask, cloth_prompt, cloth_guidance, cloth_steps],
                    outputs=cloth_output
                )

                gr.Markdown("""
                **📋 Jak stworzyć maskę:**
                1. Otwórz zdjęcie w Paint / Photoshop
                2. Zamaluj NA BIAŁO tylko ubranie
                3. Reszta niech będzie CZARNA
                4. Zapisz jako PNG
                """)

            # ================================================================
            # TAB 3: POLSKI TEKST
            # ================================================================
            with gr.Tab("✍️ Polski Tekst"):
                gr.Markdown("### Dodaj tekst z polskimi znakami!")

                with gr.Row():
                    with gr.Column():
                        text_img = gr.Image(label="🖼️ Obraz bazowy", type="filepath")
                        text_content = gr.Textbox(
                            label="📝 Tekst (POLSKIE ZNAKI OK!)",
                            placeholder="Wpisz tekst z ąćęłńóśźż...",
                            lines=3
                        )

                        with gr.Row():
                            text_x = gr.Number(label="X", value=50)
                            text_y = gr.Number(label="Y", value=50)

                        with gr.Row():
                            text_size = gr.Slider(
                                minimum=12, maximum=200, value=64, step=4,
                                label="📏 Rozmiar czcionki"
                            )

                        text_color = gr.ColorPicker(label="🎨 Kolor", value="#FFFFFF")

                        text_btn = gr.Button("✍️ DODAJ TEKST", variant="primary", size="lg")

                    with gr.Column():
                        text_output = gr.Image(label="Wynik", type="filepath")

                def add_text_wrapper(img, text, x, y, size, color):
                    # Konwertuj hex na RGB
                    color_rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    return add_polish_text(img, text, position=(int(x), int(y)),
                                          font_size=int(size), color=color_rgb)

                text_btn.click(
                    fn=add_text_wrapper,
                    inputs=[text_img, text_content, text_x, text_y, text_size, text_color],
                    outputs=text_output
                )

            # ================================================================
            # TAB 4: WATERMARK
            # ================================================================
            with gr.Tab("💧 Watermark"):
                gr.Markdown("### Dodaj znak wodny do obrazu")

                with gr.Row():
                    with gr.Column():
                        wm_img = gr.Image(label="🖼️ Obraz", type="filepath")
                        wm_text = gr.Textbox(
                            label="💧 Tekst watermarku",
                            placeholder="© 2025 Twoja Firma",
                            lines=2
                        )

                        wm_position = gr.Dropdown(
                            choices=["bottom-right", "bottom-left", "top-right", "top-left", "center"],
                            value="bottom-right",
                            label="📍 Pozycja"
                        )

                        with gr.Row():
                            wm_size = gr.Slider(
                                minimum=12, maximum=72, value=24, step=4,
                                label="📏 Rozmiar"
                            )
                            wm_opacity = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                                label="👁️ Przezroczystość"
                            )

                        wm_btn = gr.Button("💧 DODAJ WATERMARK", variant="primary", size="lg")

                    with gr.Column():
                        wm_output = gr.Image(label="Wynik", type="filepath")

                wm_btn.click(
                    fn=lambda img, text, pos, size, opa: add_watermark(
                        img, text, position=pos, font_size=int(size), opacity=opa
                    ),
                    inputs=[wm_img, wm_text, wm_position, wm_size, wm_opacity],
                    outputs=wm_output
                )

            # ================================================================
            # TAB 5: UPSCALING
            # ================================================================
            with gr.Tab("📈 Upscaling"):
                gr.Markdown("### Zwiększ rozdzielczość do 4K!")

                with gr.Row():
                    with gr.Column():
                        up_img = gr.Image(label="🖼️ Obraz do upscale", type="filepath")

                        up_method = gr.Radio(
                            choices=["ai", "fast"],
                            value="ai",
                            label="⚙️ Metoda",
                            info="AI = wolniejsze, lepsze | Fast = szybkie"
                        )

                        with gr.Row():
                            up_4x_btn = gr.Button("📈 Upscale 4x", variant="primary")
                            up_4k_btn = gr.Button("🎬 Upscale do 4K", variant="primary")

                    with gr.Column():
                        up_output = gr.Image(label="Wynik", type="filepath")

                up_4x_btn.click(
                    fn=lambda img, method: upscale_4x(img, method=method),
                    inputs=[up_img, up_method],
                    outputs=up_output
                )

                up_4k_btn.click(
                    fn=upscale_to_4k,
                    inputs=up_img,
                    outputs=up_output
                )

                gr.Markdown("""
                **⚠️ Uwagi:**
                - Upscaling AI wymaga GPU i zajmuje ~1-2 minuty
                - 4K (3840x2160) wymaga ~8GB RAM
                - Metoda "fast" działa natychmiast
                """)

            # ================================================================
            # TAB 6: ENHANCE (Poprawa jakości)
            # ================================================================
            with gr.Tab("✨ Enhance"):
                gr.Markdown("### Szybka poprawa jakości zdjęcia")

                with gr.Row():
                    with gr.Column():
                        enh_img = gr.Image(label="🖼️ Zdjęcie", type="filepath")

                        enh_sharp = gr.Slider(
                            minimum=0.5, maximum=2.0, value=1.3, step=0.1,
                            label="🔍 Ostrość"
                        )
                        enh_color = gr.Slider(
                            minimum=0.5, maximum=2.0, value=1.2, step=0.1,
                            label="🎨 Saturacja kolorów"
                        )
                        enh_contrast = gr.Slider(
                            minimum=0.5, maximum=2.0, value=1.1, step=0.1,
                            label="◐ Kontrast"
                        )
                        enh_bright = gr.Slider(
                            minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                            label="☀️ Jasność"
                        )

                        enh_btn = gr.Button("✨ POPRAW JAKOŚĆ", variant="primary", size="lg")

                    with gr.Column():
                        enh_output = gr.Image(label="Wynik", type="filepath")

                enh_btn.click(
                    fn=lambda img, sharp, color, cont, bright: enhance_photo(
                        img, sharpness=sharp, color=color,
                        contrast=cont, brightness=bright
                    ),
                    inputs=[enh_img, enh_sharp, enh_color, enh_contrast, enh_bright],
                    outputs=enh_output
                )

        # Footer
        gr.Markdown("""
        ---
        ### 🎨 ALPHA IMAGE STUDIO v1.0
        Stworzone z ❤️ | Powered by Stable Diffusion & Gradio
        """)

    return app


def launch():
    """Uruchom aplikację"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║              🎨 ALPHA IMAGE STUDIO 🎨                            ║
    ║                                                                  ║
    ║  Uruchamianie interfejsu graficznego...                          ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    app = create_ui()

    # Uruchom serwer
    app.launch(
        server_name="0.0.0.0",  # Dostępne z sieci
        server_port=7860,
        share=False,  # Zmień na True dla publicznego linku
        show_error=True
    )


if __name__ == "__main__":
    launch()
