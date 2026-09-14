from paddleocr import PaddleOCR

# Initialize (new API)
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang='en'
)

img_path = "sample/image2.jpg"   # ← change this to your actual image path

# New method is predict()
result = ocr.predict(img_path)

print("==== OCR Result ====\n")

for res in result:
    # Print nicely
    res.print()

    # Also extract just the text
    print("\n==== Extracted Texts ====")
    if hasattr(res, 'json') and res.json:
        data = res.json
        # Try to get texts
        if 'rec_texts' in data:
            for text in data['rec_texts']:
                print(text)
        elif 'res' in data and 'rec_texts' in data['res']:
            for text in data['res']['rec_texts']:
                print(text)
