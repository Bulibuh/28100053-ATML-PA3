"""Render operational Markdown to a readable PDF; never generate a research report."""

from html import escape
from pathlib import Path
import textwrap
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    font_root = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, filename in [
        ("GuideSans", "DejaVuSans.ttf"),
        ("GuideBold", "DejaVuSans-Bold.ttf"),
        ("GuideMono", "DejaVuSansMono.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont(name, str(font_root / filename)))
    styles = getSampleStyleSheet()
    styles["Title"].fontName = "GuideBold"
    styles["Title"].fontSize = 17
    styles["Heading2"].fontName = "GuideBold"
    styles["Heading2"].fontSize = 11.5
    styles.add(
        ParagraphStyle(
            name="BodyGuide", fontName="GuideSans", fontSize=9, leading=13, spaceAfter=8
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeGuide",
            fontName="GuideMono",
            fontSize=7.5,
            leading=10,
            leftIndent=8,
            rightIndent=8,
            backColor=colors.HexColor("#f1f4f8"),
            spaceBefore=6,
            spaceAfter=10,
        )
    )
    path = ROOT / "docs/ATML_PA3_Colab_Commands.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=42,
        bottomMargin=40,
        title="ATML PA3 — Colab A100 Commands",
        author="ATML PA3 code repository",
    )
    story = []
    code = False
    lines = []
    paragraph = []

    def flush():
        if paragraph:
            story.append(Paragraph(escape(" ".join(paragraph)), styles["BodyGuide"]))
            paragraph.clear()

    for line in (ROOT / "docs/COLAB_COMMANDS.md").read_text().splitlines():
        if line.startswith("```"):
            flush()
            if code:
                # Long code wraps visually with an indentation marker, preserving the Markdown source as authoritative.
                wrapped = []
                for value in lines:
                    wrapped.extend(
                        textwrap.wrap(
                            value,
                            width=103,
                            subsequent_indent="    ",
                            replace_whitespace=False,
                            drop_whitespace=False,
                            break_long_words=False,
                            break_on_hyphens=False,
                        )
                        or [""]
                    )
                story.append(
                    KeepTogether(
                        [Preformatted("\n".join(wrapped), styles["CodeGuide"])]
                    )
                )
                lines = []
            code = not code
            continue
        if code:
            lines.append(line)
            continue
        if not line.strip():
            flush()
            continue
        if line.startswith("# "):
            flush()
            story.append(Paragraph(escape(line[2:]), styles["Title"]))
        elif line.startswith("## "):
            flush()
            story.append(Spacer(1, 8))
            story.append(Paragraph(escape(line[3:]), styles["Heading2"]))
        else:
            paragraph.append(line)
    flush()

    def footer(canvas, doc):
        canvas.setFont("GuideSans", 8)
        canvas.setFillColor(colors.gray)
        canvas.drawString(40, 23, "Operational command guide — not the graded report")
        canvas.drawRightString(A4[0] - 40, 23, f"{doc.page}")

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(path)


if __name__ == "__main__":
    main()
