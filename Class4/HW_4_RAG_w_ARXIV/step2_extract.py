import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []

    for page in doc:
        # Extract raw text
        text = page.get_text("text")

        # OCR for images
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n < 5:  # grayscale/RGB
                img_bytes = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_bytes))
                text += "\n" + pytesseract.image_to_string(image)
            pix = None

        pages.append(text)

    return "\n".join(pages)

if __name__ == "__main__":
    print(extract_text_from_pdf("pdfs/sample.pdf")[:500])
