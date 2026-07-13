import os
import platform
from langchain_core.documents import Document

if platform.system() == "Windows":
    POPPLER_PATH = r"C:\Users\Karthik\Downloads\poppler\poppler-25.07.0\Library\bin"
else:
    POPPLER_PATH = None


def _has_tesseract():
    try:
        import pytesseract
        from shutil import which
        return which("tesseract") is not None
    except ImportError:
        return False


def _ocr_placeholder(page_number, pdf_path, reason):
    return Document(
        page_content=(
            f"This scanned PDF page could not be OCR'd because {reason}. "
            "Upload a text-based PDF or install Tesseract/Poppler for full OCR support."
        ),
        metadata={"page": page_number, "source": pdf_path},
    )

def load_pdf(pdf_path):
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text = "".join(doc.page_content for doc in documents)

    # Text PDF
    if len(text.strip()) > 100:
        return documents, False

    print("Scanned PDF detected. Evaluating OCR availability...")

    if not _has_tesseract():
        print("Tesseract not installed or unavailable. Skipping OCR.")
        return [
            _ocr_placeholder(
                1,
                pdf_path,
                "Tesseract is not installed or not found in PATH",
            )
        ], True

    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception as exc:
        print("OCR dependencies unavailable:", exc)
        return [
            _ocr_placeholder(
                1,
                pdf_path,
                "OCR dependencies are unavailable",
            )
        ], True

    try:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    except Exception as exc:
        print("Failed to convert PDF pages to images for OCR:", exc)
        return [
            _ocr_placeholder(
                1,
                pdf_path,
                "Poppler is not installed or not configured",
            )
        ], True

    ocr_docs = []

    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        if not text.strip():
            ocr_docs.append(
                _ocr_placeholder(
                    i + 1,
                    pdf_path,
                    "OCR produced no text for this page",
                )
            )
        else:
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