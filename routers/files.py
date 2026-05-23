import io
import logging
import re
import pdfplumber
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from auth import get_current_user

router = APIRouter(prefix="/files", tags=["files"])
logger = logging.getLogger(__name__)

MAX_PDF_SIZE = 10 * 1024 * 1024  # 10 MB


def _ocr_pdf(content: bytes) -> str:
    try:
        import fitz
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return ""

    ocr = RapidOCR()
    doc = fitz.open(stream=content, filetype="pdf")
    pages_text = []
    for page in doc:
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        result, _ = ocr(img)
        if result:
            pages_text.append(" ".join(r[1] for r in result))
    return "\n\n".join(pages_text).strip()


@router.post("/extract-text")
async def extract_text(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user),
):
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    if len(content) > MAX_PDF_SIZE:
        raise HTTPException(status_code=413, detail="PDF too large. Maximum size is 10 MB.")

    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages = len(pdf.pages)
            text = "\n\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()
    except Exception as exc:
        logger.exception("pdfplumber failed to parse upload: %s", exc)
        raise HTTPException(status_code=422, detail="Could not extract text from this PDF.")

    if not text:
        text = _ocr_pdf(content)

    if not text:
        raise HTTPException(status_code=422, detail="Could not extract text from this PDF.")

    return {"text": text, "pages": pages}
