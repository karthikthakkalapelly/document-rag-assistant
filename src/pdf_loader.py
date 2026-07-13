import base64
import io
import os
import platform
from typing import List
from langchain_core.documents import Document
import requests

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


def _get_google_api_key():
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def _ocr_placeholder(page_number, pdf_path, reason):
    return Document(
        page_content=(
            f"This scanned PDF page could not be OCR'd because {reason}. "
            "Upload a text-based PDF or install Tesseract/Poppler for full OCR support."
        ),
        metadata={"page": page_number, "source": pdf_path},
    )


def _render_pdf_pages(pdf_path):
    from pdf2image import convert_from_path

    return convert_from_path(pdf_path, poppler_path=POPPLER_PATH)


def _extract_images_from_pdf(pdf_path) -> List[object]:
    from pypdf import PdfReader
    from PIL import Image

    images = []
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        resources = page.get("/Resources")
        if resources is None:
            continue
        xobjects = resources.get("/XObject")
        if xobjects is None:
            continue
        xobjects = xobjects.get_object()
        for obj in xobjects.values():
            obj = obj.get_object()
            if obj.get("/Subtype") != "/Image":
                continue
            try:
                data = obj.get_data()
                image = Image.open(io.BytesIO(data))
                images.append(image.convert("RGB"))
            except Exception:
                continue
    return images


def _google_vision_ocr(image, api_key):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    payload = {
        "requests": [
            {
                "image": {"content": image_base64},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            }
        ]
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        print("Google Vision OCR request failed:", exc)
        return ""
    except requests.exceptions.RequestException as exc:
        print("Google Vision OCR network error:", exc)
        return ""

    result = response.json()
    annotations = result.get("responses", [{}])[0].get("fullTextAnnotation", {})
    return annotations.get("text", "").strip()


def _ocr_image(image, api_key=None):
    if _has_tesseract():
        import pytesseract
        return pytesseract.image_to_string(image).strip()

    if api_key:
        return _google_vision_ocr(image, api_key)

    return ""


def load_pdf(pdf_path):
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text = "".join(doc.page_content for doc in documents)

    if len(text.strip()) > 100:
        return documents, False

    print("Scanned PDF detected. Running OCR fallback...")
    api_key = _get_google_api_key()

    images = []
    try:
        images = _render_pdf_pages(pdf_path)
    except Exception as exc:
        print("PDF rendering failed, trying image extraction:", exc)
        try:
            images = _extract_images_from_pdf(pdf_path)
        except Exception as exc2:
            print("PDF image extraction failed:", exc2)
            images = []

    if not images:
        return [
            _ocr_placeholder(
                1,
                pdf_path,
                "the PDF could not be converted to images on this platform",
            )
        ], True

    ocr_docs = []
    for page_number, image in enumerate(images, start=1):
        ocr_text = _ocr_image(image, api_key=api_key)
        if not ocr_text:
            reason = (
                "OCR failed with the available extractor"
                if api_key or _has_tesseract()
                else "Tesseract is not installed and no Google API key is configured"
            )
            ocr_docs.append(_ocr_placeholder(page_number, pdf_path, reason))
        else:
            ocr_docs.append(
                Document(
                    page_content=ocr_text,
                    metadata={"page": page_number, "source": pdf_path},
                )
            )

    return ocr_docs, True
