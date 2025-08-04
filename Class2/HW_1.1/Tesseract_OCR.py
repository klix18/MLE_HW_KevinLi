from PIL import Image
import pytesseract  # Make sure pytesseract is installed

# Load an image using Pillow (PIL)
image = Image.open('/Users/kevinli_home/Desktop/MLE_in_Gen_AI-Course/class2/03_Demo/HW_1.1/input1_shakespear-text.jpg')

# Perform OCR on the image
text = pytesseract.image_to_string(image)


# Export the text to a .txt file
output_path = '/Users/kevinli_home/Desktop/MLE_in_Gen_AI-Course/class2/03_Demo/HW_1.1/output_text.txt'
with open(output_path, 'w') as file:
    file.write(text)

print(f"OCR result saved to {output_path}")
