%%writefile preprocessing.py
import os
import re
import string
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd

def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.txt':
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    elif ext == '.pdf':
        reader = PdfReader(filepath)
        return ' '.join(page.extract_text() for page in reader.pages)
    elif ext == '.docx':
        doc = Document(filepath)
        return '\n'.join([para.text for para in doc.paragraphs])
    elif ext == '.csv':
        df = pd.read_csv(filepath)
        return df.to_string()
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = '\n'.join([re.sub(r'\s+', ' ', line) for line in text.splitlines()])
    return text

def segment_text(text):
    paragraphs = text.split('\n')
    return [para.strip() for para in paragraphs if para.strip()]

def preprocess_document(filepath):
    try:
        raw_text = extract_text(filepath)
        normalized_text = normalize_text(raw_text)
        paragraphs = segment_text(normalized_text)
        return paragraphs
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return []
