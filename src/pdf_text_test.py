from pdfminer.high_level import extract_text
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "ACC_Sample.pdf")

text = extract_text(pdf_path)

print(text[1000:3000])  # print first 3000 characters
