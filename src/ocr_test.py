from pdf2image import convert_from_path
import pytesseract

pages = convert_from_path("ACC_Sample.pdf", dpi=300)

for i, page in enumerate(pages):
    page.save(f"page_{i}.jpg", "JPEG")
    text = pytesseract.image_to_string(f"page_{i}.jpg")
    print(f"\n--- PAGE {i} ---\n")
    print(text)
