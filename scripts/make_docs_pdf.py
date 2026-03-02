from __future__ import annotations
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

SRC = Path('docs/src')
OUT = Path('docs/build/MMx_Enterprise_System_Manual_v3.0.pdf')

def build() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=A4)
    width, height = A4
    y = height - 60
    c.setFont('Helvetica-Bold', 16)
    c.drawString(50, y, 'MMx Enterprise System Manual v3.0')
    y -= 30
    c.setFont('Helvetica', 10)
    for md in ['architecture.md','metric_dictionary.md','operations_manual.md','technical_appendix.md','program_inventory.md']:
        p = SRC / md
        if not p.exists():
            continue
        c.setFont('Helvetica-Bold', 12)
        c.drawString(50, y, md)
        y -= 18
        c.setFont('Helvetica', 9)
        for line in p.read_text(encoding='utf-8').splitlines():
            if y < 60:
                c.showPage()
                y = height - 60
                c.setFont('Helvetica', 9)
            c.drawString(55, y, line[:120])
            y -= 12
        y -= 12
    c.save()

if __name__ == '__main__':
    build()
    print('written:', OUT)
