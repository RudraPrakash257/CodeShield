"""Plain text file extraction."""
from pathlib import Path


def extract_text_from_file(file_path: str) -> dict:
    """
    Extract text from TXT/MD files.

    Returns:
        dict with filename, text, pages, is_handwritten, ocr_used
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        return {
            "filename": Path(file_path).name,
            "text": text,
            "pages": max(1, len(text.split('\n')) // 30),  # Rough estimate
            "is_handwritten": False,
            "ocr_used": False
        }

    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()

            return {
                "filename": Path(file_path).name,
                "text": text,
                "pages": max(1, len(text.split('\n')) // 30),
                "is_handwritten": False,
                "ocr_used": False
            }
        except Exception as e:
            raise Exception(f"Failed to read text file: {str(e)}")

    except Exception as e:
        raise Exception(f"Failed to extract text: {str(e)}")
