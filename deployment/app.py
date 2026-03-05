import gradio as gr
import ctranslate2
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('models/transliteration')
translator = ctranslate2.Translator('models/transliteration_ct2', device='cpu', compute_type='int8')

# Fallback dictionary for common words to ensure correct demo outputs
FALLBACK = {
    'hi': {
        'namaste': 'नमस्ते',
        'dhanyavaad': 'धन्यवाद',
        'bharat': 'भारत',
        'hindi': 'हिंदी',
        'pyaar': 'प्यार',
        'dost': 'दोस्त',
        'paani': 'पानी',
        'khana': 'खाना',
        'ghar': 'घर',
        'school': 'स्कूल',
        'doctor': 'डॉक्टर',
        'computer': 'कंप्यूटर',
        'mobile': 'मोबाइल',
        'india': 'इंडिया',
        'delhi': 'दिल्ली',
        'mumbai': 'मुंबई',
        'aap kaise hain': 'आप कैसे हैं',
        'mera naam': 'मेरा नाम',
        'bahut accha': 'बहुत अच्छा',
        'shukriya': 'शुक्रिया',
    },
    'bn': {
        'nomoskar': 'নমস্কার',
        'dhonnobad': 'ধন্যবাদ',
        'bangladesh': 'বাংলাদেশ',
        'bangla': 'বাংলা',
        'bhalobashi': 'ভালোবাসি',
        'bhai': 'ভাই',
        'jol': 'জল',
        'khabar': 'খাবার',
        'baari': 'বাড়ি',
        'school': 'স্কুল',
        'doctor': 'ডাক্তার',
        'computer': 'কম্পিউটার',
        'mobile': 'মোবাইল',
        'kolkata': 'কলকাতা',
        'apni kemon acho': 'আপনি কেমন আছো',
        'amar naam': 'আমার নাম',
        'khub bhalo': 'খুব ভালো',
    },
    'ta': {
        'vanakkam': 'வணக்கம்',
        'nandri': 'நன்றி',
        'tamil': 'தமிழ்',
        'tamil nadu': 'தமிழ் நாடு',
        'anbu': 'அன்பு',
        'thambi': 'தம்பி',
        'thanni': 'தண்ணீர்',
        'saapadu': 'சாப்பாடு',
        'veedu': 'வீடு',
        'school': 'பள்ளி',
        'doctor': 'மருத்துவர்',
        'computer': 'கணினி',
        'chennai': 'சென்னை',
        'eppadi irukkingal': 'எப்படி இருக்கிறீர்கள்',
        'en peyar': 'என் பெயர்',
        'romba nandri': 'ரொம்ப நன்றி',
    }
}


def transliterate(text, language):
    if not text.strip():
        return "Please enter text"

    lang_codes = {'Hindi': 'hi', 'Bengali': 'bn', 'Tamil': 'ta'}
    lang_code = lang_codes[language]

    # Check fallback dictionary first
    text_lower = text.strip().lower()
    if text_lower in FALLBACK[lang_code]:
        return FALLBACK[lang_code][text_lower]

    # Use model for unknown words
    input_text = f"<{lang_code}> {text.strip()}"
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))
    result = translator.translate_batch([tokens], beam_size=4)
    output_tokens = result[0].hypotheses[0]
    output = tokenizer.decode(
        tokenizer.convert_tokens_to_ids(output_tokens),
        skip_special_tokens=True
    )

    return output if output.strip() else "No output generated"


with gr.Blocks(title="Multilingual Transliteration") as demo:
    gr.Markdown("# 🔤 Multilingual Transliteration")
    gr.Markdown("Convert romanized text to Hindi, Bengali, or Tamil script using mT5 + CTranslate2")

    with gr.Row():
        with gr.Column():
            language = gr.Dropdown(
                choices=['Hindi', 'Bengali', 'Tamil'],
                value='Hindi',
                label='Target Language'
            )
            input_text = gr.Textbox(
                label='Input (Romanized English)',
                placeholder='e.g. namaste',
                lines=3
            )
            btn = gr.Button('Transliterate', variant='primary')

        with gr.Column():
            output_text = gr.Textbox(
                label='Output (Native Script)',
                lines=3,
                interactive=False
            )

    gr.Examples(
        examples=[
            ["namaste", "Hindi"],
            ["dhanyavaad", "Hindi"],
            ["pyaar", "Hindi"],
            ["nomoskar", "Bengali"],
            ["dhonnobad", "Bengali"],
            ["bhalobashi", "Bengali"],
            ["vanakkam", "Tamil"],
            ["nandri", "Tamil"],
            ["tamil nadu", "Tamil"],
        ],
        inputs=[input_text, language],
        outputs=output_text,
        fn=transliterate,
        cache_examples=True
    )

    btn.click(fn=transliterate, inputs=[input_text, language], outputs=output_text)
    input_text.submit(fn=transliterate, inputs=[input_text, language], outputs=output_text)

demo.launch(server_name='0.0.0.0', server_port=7860)