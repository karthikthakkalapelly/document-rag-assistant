import platform

if platform.system() == "Windows":
    POPPLER_PATH = r"C:\Users\Karthik\Downloads\poppler\poppler-25.07.0\Library\bin"
else:
    POPPLER_PATH = None


def load_pdf(pdf_path):
    from langchain_community.document_loaders import PyPDFLoader
    from pdf2image import convert_from_path
    import pytesseract
    from langchain_core.documents import Document

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text = "".join(doc.page_content for doc in documents)

    # Text PDF
    if len(text.strip()) > 100:
        return documents, False

    print("Scanned PDF detected. Running OCR...")

    images = convert_from_path(
        pdf_path,
        poppler_path=POPPLER_PATH
    )

    ocr_docs = []

    for i, image in enumerate(images):

        text = pytesseract.image_to_string(image)

        ocr_docs.append(
            Document(
                page_content=text,
                metadata={
                    "page": i + 1,
                    "source": pdf_path
                }
            )
        )

    return ocr_docs, True