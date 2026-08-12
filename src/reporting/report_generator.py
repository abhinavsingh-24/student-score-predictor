from pathlib import Path
import sys
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

src_root = Path(__file__).resolve()
while src_root.name != "src" and src_root.parent != src_root:
    src_root = src_root.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from utils import get_logger

logger = get_logger(__name__)


def generate_pdf_report(output_pdf: Path, title: str, metrics: dict, image_paths: list = None):
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_pdf), pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 60, title)
    c.setFont("Helvetica", 12)
    y = height - 90
    c.drawString(40, y, "Metrics:")
    y -= 20
    for k, v in metrics.items():
        c.drawString(60, y, f"{k}: {v}")
        y -= 16
    if image_paths:
        for img in image_paths:
            try:
                c.drawImage(str(img), 40, y - 200, width=500, height=200)
                y -= 220
            except Exception as e:
                logger.warning(f"Could not add image {img} to report: {e}")
    c.showPage()
    c.save()
    logger.info(f"Saved PDF report to {output_pdf}")
    return output_pdf
