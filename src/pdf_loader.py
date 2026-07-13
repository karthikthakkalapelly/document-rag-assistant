import io
import os
import platform
from typing import List
from langchain_core.documents import Document

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
    raise RuntimeError("Google Vision OCR has been removed. Use Tesseract (pytesseract) instead.")


def _ocr_image(image):
    """OCR a PIL image using pytesseract. Returns extracted text or empty string on failure.

    Does not call external cloud OCR. Errors are logged and an empty string is returned
    so the caller can attach a placeholder document instead of crashing.
    """
    tesseract_available = _has_tesseract()
    print("OCR: starting image OCR. Tesseract available:", tesseract_available)
    if not tesseract_available:
        print("OCR: pytesseract not available; skipping OCR")
        return ""

    try:
        import pytesseract
        text = pytesseract.image_to_string(image).strip()
        print("OCR: pytesseract produced", len(text), "chars")
        return text
    except Exception as e:
        print("pytesseract OCR failed:", type(e).__name__, e)
        # Return empty string so caller can attach a placeholder and continue
        return ""


def load_pdf(pdf_path):
    from langchain_community.document_loaders import PyPDFLoader
    print("Loading PDF via PyPDFLoader:", pdf_path)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text = "".join(doc.page_content for doc in documents)
    print("PyPDFLoader produced", len(documents), "pages with total text length", len(text))

    if len(text.strip()) > 100:
        return documents, False

    print("Scanned PDF detected. Running OCR fallback using pytesseract...")

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
            # return placeholder indicating conversion failed
            return [
                _ocr_placeholder(
                    1,
                    pdf_path,
                    "the PDF could not be converted to images on this platform",
                )
            ], True

    if not images:
        raise RuntimeError("No images found/rendered from PDF for OCR")

    ocr_docs = []
    for page_number, image in enumerate(images, start=1):
        print(f"OCR: processing page {page_number}/{len(images)}")
        ocr_text = _ocr_image(image)
        print(f"OCR: page {page_number} text length: {len(ocr_text)}")
        if not ocr_text:
            reason = (
                "Tesseract is not installed or OCR failed on this page."
                " Install Tesseract to enable OCR."
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
