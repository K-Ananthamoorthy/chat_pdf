from flask import Flask, request, jsonify, render_template
import os
import fitz  # PyMuPDF
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
os.makedirs('uploads', exist_ok=True)

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def chat_with_gpt2(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files['file']
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        text = extract_text_from_pdf(file_path)
        return jsonify({'text': text})
    return "No file uploaded", 400

@app.route('/chat', methods=['POST'])
def chat():
    prompt = request.json.get('prompt')
    if prompt:
        response = chat_with_gpt2(prompt)
        return jsonify({'response': response})
    return "No prompt provided", 400

if __name__ == '__main__':
    app.run(debug=True)
