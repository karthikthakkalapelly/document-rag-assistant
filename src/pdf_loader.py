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
        tpath = which("tesseract")
        if tpath:
            print("Tesseract binary found at:", tpath)
        else:
            print("Tesseract binary not found in PATH")
        return tpath is not None
    except ImportError:
        print("pytesseract not installed")
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
    print("Rendering PDF pages with pdf2image (poppler_path=", POPPLER_PATH, ")")
    try:
        if POPPLER_PATH and os.path.exists(POPPLER_PATH):
            images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        else:
            images = convert_from_path(pdf_path)
        print(f"pdf2image rendered {len(images)} pages")
        return images
    except Exception as e:
        print("pdf2image failed:", type(e).__name__, e)
        print("Attempting PyMuPDF (fitz) fallback to render pages")
        try:
            import fitz  # PyMuPDF
            from PIL import Image

            images = []
            doc = fitz.open(pdf_path)
            for page in doc:
                pix = page.get_pixmap(alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            print(f"PyMuPDF rendered {len(images)} pages")
            return images
        except Exception as e2:
            print("PyMuPDF fallback failed:", type(e2).__name__, e2)
            # propagate the original pdf2image exception for visibility
            raise


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
        # propagate so caller can surface the error
        raise
    except requests.exceptions.RequestException as exc:
        print("Google Vision OCR network error:", exc)
        raise

    result = response.json()
    annotations = result.get("responses", [{}])[0].get("fullTextAnnotation", {})
    return annotations.get("text", "").strip()


def _ocr_image(image, api_key=None):
    print("OCR: starting image OCR. Tesseract available:", _has_tesseract(), "Google API key present:", bool(api_key))
    if _has_tesseract():
        try:
            import pytesseract
            text = pytesseract.image_to_string(image).strip()
            print("OCR: pytesseract produced", len(text), "chars")
            return text
        except Exception as e:
            print("pytesseract OCR failed:", type(e).__name__, e)
            # propagate the exception so the failure is visible
            raise

    if api_key:
        # _google_vision_ocr will raise on HTTP/network errors
        text = _google_vision_ocr(image, api_key)
        print("OCR: Google Vision produced", len(text), "chars")
        return text

    raise RuntimeError("No OCR engine available: install Tesseract or set GOOGLE_API_KEY")


def load_pdf(pdf_path):
    from langchain_community.document_loaders import PyPDFLoader
    print("Loading PDF via PyPDFLoader:", pdf_path)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text = "".join(doc.page_content for doc in documents)
    print("PyPDFLoader produced", len(documents), "pages with total text length", len(text))

    if len(text.strip()) > 100:
        return documents, False

    print("Scanned PDF detected. Running OCR fallback...")
    api_key = _get_google_api_key()

    images = []
    try:
        images = _render_pdf_pages(pdf_path)
    except Exception as exc:
        print("PDF rendering failed:", type(exc).__name__, exc)
        print("Trying to extract embedded images from PDF pages")
        try:
            images = _extract_images_from_pdf(pdf_path)
            print(f"Extracted {len(images)} embedded images from PDF")
        except Exception as exc2:
            print("PDF image extraction failed:", type(exc2).__name__, exc2)
            # propagate so the caller/Streamlit shows the error
            raise

    if not images:
        raise RuntimeError("No images found/rendered from PDF for OCR")

    ocr_docs = []
    for page_number, image in enumerate(images, start=1):
        print(f"OCR: processing page {page_number}/{len(images)}")
        try:
            ocr_text = _ocr_image(image, api_key=api_key)
        except Exception:
            print(f"OCR failed on page {page_number}; attaching placeholder")
            raise

        print(f"OCR: page {page_number} text length: {len(ocr_text)}")
        if not ocr_text:
            reason = (
                "OCR produced no text"
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
