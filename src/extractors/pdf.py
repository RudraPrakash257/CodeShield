"""PDF text extraction."""
import pypdf
from pathlib import Path


def extract_text_from_pdf(file_path: str) -> dict:
    """
    Extract text from PDF file.

    Returns:
        dict with filename, text, pages, is_handwritten, ocr_used
    """
    try:
        reader = pypdf.PdfReader(file_path)
        text_parts = []
        pages = len(reader.pages)

        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

        full_text = "\n".join(text_parts).strip()

        # If extracted text is very short or empty, it's likely scanned/handwritten
        is_handwritten = len(full_text) < 50

        return {
            "filename": Path(file_path).name,
            "text": full_text,
            "pages": pages,
            "is_handwritten": is_handwritten,
            "ocr_used": False
        }

    except Exception as e:
        raise Exception(f"Failed to extract PDF: {str(e)}")
