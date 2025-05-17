import os
import PyPDF2
import docx

def convert_pdf_to_text(pdf_path):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
            
            base_name = os.path.splitext(pdf_path)[0]
            text_file_path = base_name + ".txt"
            
            with open(text_file_path, 'w', encoding='utf-8') as text_file:
                text_file.write(text)
            print(f"Successfully converted {pdf_path} to {text_file_path}")
    except Exception as e:
        print(f"Error converting {pdf_path}: {e}")

def convert_docx_to_text(docx_path):
    try:
        doc = docx.Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
            
        base_name = os.path.splitext(docx_path)[0]
        text_file_path = base_name + ".txt"
        
        with open(text_file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)
        print(f"Successfully converted {docx_path} to {text_file_path}")
    except Exception as e:
        print(f"Error converting {docx_path}: {e}")

if __name__ == "__main__":
    current_directory = os.getcwd()
    for filename in os.listdir(current_directory):
        file_path = os.path.join(current_directory, filename)
        if filename.lower().endswith(".pdf"):
            convert_pdf_to_text(file_path)
        elif filename.lower().endswith(".docx"):
            convert_docx_to_text(file_path)
    print("Conversion process completed.")
