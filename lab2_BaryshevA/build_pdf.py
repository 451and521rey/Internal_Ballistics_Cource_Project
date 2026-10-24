"""Сборка report.pdf из report.md по сокращённому профилю ГОСТ.

Парсер намеренно поддерживает только конструкции, используемые в отчёте этой
лабораторной работы. Поиск и изменение численных результатов здесь отсутствуют.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPORT_MD = PROJECT_DIR / "report.md"
REPORT_PDF = PROJECT_DIR / "report.pdf"


def _register_fonts() -> tuple[str, str, str]:
    windows_fonts = Path("C:/Windows/Fonts")
    regular = windows_fonts / "times.ttf"
    bold = windows_fonts / "timesbd.ttf"
    italic = windows_fonts / "timesi.ttf"
    if not (regular.exists() and bold.exists() and italic.exists()):
        from matplotlib import font_manager

        regular = Path(font_manager.findfont("DejaVu Serif"))
        bold = Path(
            font_manager.findfont(
                font_manager.FontProperties(family="DejaVu Serif", weight="bold")
            )
        )
        italic = Path(
            font_manager.findfont(
                font_manager.FontProperties(family="DejaVu Serif", style="italic")
            )
        )
    pdfmetrics.registerFont(TTFont("ReportSerif", str(regular)))
    pdfmetrics.registerFont(TTFont("ReportSerif-Bold", str(bold)))
    pdfmetrics.registerFont(TTFont("ReportSerif-Italic", str(italic)))
    pdfmetrics.registerFontFamily(
        "ReportSerif",
        normal="ReportSerif",
        bold="ReportSerif-Bold",
        italic="ReportSerif-Italic",
        boldItalic="ReportSerif-Bold",
    )
    return "ReportSerif", "ReportSerif-Bold", "ReportSerif-Italic"


def _styles(font: str, bold_font: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "BodyGost",
            parent=base["Normal"],
            fontName=font,
            fontSize=14,
            leading=21,
            alignment=TA_JUSTIFY,
            firstLineIndent=12.5 * mm,
            spaceBefore=0,
            spaceAfter=0,
            allowWidows=0,
            allowOrphans=0,
        ),
        "section": ParagraphStyle(
            "SectionGost",
            fontName=bold_font,
            fontSize=14,
            leading=18,
            alignment=TA_LEFT,
            firstLineIndent=12.5 * mm,
            spaceAfter=8 * mm,
            keepWithNext=True,
        ),
        "structural": ParagraphStyle(
            "StructuralGost",
            fontName=bold_font,
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=8 * mm,
            keepWithNext=True,
        ),
        "subsection": ParagraphStyle(
            "SubsectionGost",
            fontName=bold_font,
            fontSize=14,
            leading=18,
            alignment=TA_LEFT,
            firstLineIndent=12.5 * mm,
            spaceBefore=5 * mm,
            spaceAfter=3 * mm,
            keepWithNext=True,
        ),
        "formula": ParagraphStyle(
            "FormulaGost",
            fontName=font,
            fontSize=14,
            leading=20,
            alignment=TA_CENTER,
            spaceBefore=3 * mm,
            spaceAfter=3 * mm,
        ),
        "table_caption": ParagraphStyle(
            "TableCaptionGost",
            fontName=font,
            fontSize=12,
            leading=15,
            alignment=TA_LEFT,
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
            keepWithNext=True,
        ),
        "figure_caption": ParagraphStyle(
            "FigureCaptionGost",
            fontName=font,
            fontSize=12,
            leading=15,
            alignment=TA_CENTER,
            spaceBefore=2 * mm,
            spaceAfter=4 * mm,
        ),
        "table_cell": ParagraphStyle(
            "TableCellGost",
            fontName=font,
            fontSize=8.5,
            leading=10.5,
            alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "TableHeaderGost",
            fontName=bold_font,
            fontSize=8.5,
            leading=10.5,
            alignment=TA_CENTER,
        ),
        "code": ParagraphStyle(
            "CodeGost",
            fontName=font,
            fontSize=9,
            leading=11,
            alignment=TA_LEFT,
            leftIndent=5 * mm,
            rightIndent=5 * mm,
            spaceBefore=2 * mm,
            spaceAfter=2 * mm,
        ),
        "title_center": ParagraphStyle(
            "TitleCenter",
            fontName=font,
            fontSize=14,
            leading=21,
            alignment=TA_CENTER,
        ),
        "title_bold": ParagraphStyle(
            "TitleBold",
            fontName=bold_font,
            fontSize=16,
            leading=22,
            alignment=TA_CENTER,
        ),
        "title_right": ParagraphStyle(
            "TitleRight",
            fontName=font,
            fontSize=14,
            leading=21,
            alignment=TA_RIGHT,
        ),
    }


def _latex_to_text(value: str) -> str:
    text = value.strip()
    tag = re.search(r"\\tag\{([^}]+)\}", text)
    text = re.sub(r"\\tag\{[^}]+\}", "", text)
    replacements = {
        r"\left": "",
        r"\right": "",
        r"\qquad": "    ",
        r"\quad": "  ",
        r"\;": " ",
        r"\,": " ",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\times": "×",
        r"\cdot": "·",
        r"\rho": "rho",
        r"\nabla": "grad ",
        r"\sum": "Σ",
        r"\cos": "cos",
        r"\sin": "sin",
        r"\max": "max",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    fraction_pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while fraction_pattern.search(text):
        text = fraction_pattern.sub(r"(\1)/(\2)", text)
    text = text.replace("^2", "²").replace("^3", "³")
    text = text.replace("\\", " ").replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text).strip()
    if tag:
        text += f"    ({tag.group(1)})"
    return text


def _inline_markup(value: str) -> str:
    links: list[tuple[str, str]] = []

    def save_link(match: re.Match[str]) -> str:
        links.append((match.group(1), match.group(2)))
        return f"@@LINK{len(links) - 1}@@"

    value = re.sub(r"\[([^]]+)\]\(([^)]+)\)", save_link, value)
    value = re.sub(r"\$([^$]+)\$", lambda match: _latex_to_text(match.group(1)), value)
    value = html.escape(value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`([^`]+)`", r"<font name='ReportSerif-Italic'>\1</font>", value)
    for index, (label, url) in enumerate(links):
        replacement = f"{html.escape(label)} ({html.escape(url)})"
        value = value.replace(f"@@LINK{index}@@", replacement)
    return value


def _title_page(styles: dict[str, ParagraphStyle]) -> list[Any]:
    return [
        Paragraph(
            "Московский государственный технический университет<br/>"
            "имени Н. Э. Баумана",
            styles["title_center"],
        ),
        Paragraph("Кафедра СМ6 «Ракетные и импульсные системы»", styles["title_center"]),
        Paragraph("Дисциплина «Методы оптимизации»", styles["title_center"]),
        Spacer(1, 43 * mm),
        Paragraph("ОТЧЁТ ПО ЛАБОРАТОРНОЙ РАБОТЕ № 2", styles["title_bold"]),
        Spacer(1, 8 * mm),
        Paragraph(
            "«Методы оптимизации: SciPy, Optuna<br/>"
            "и проверяемое использование LLM»",
            styles["title_bold"],
        ),
        Spacer(1, 42 * mm),
        Paragraph("Выполнил: студент группы СМ6-72<br/>Барышев А. А.", styles["title_right"]),
        Spacer(1, 5 * mm),
        Paragraph("Преподаватель: Товарнов М. С.", styles["title_right"]),
        Spacer(1, 42 * mm),
        Paragraph("Москва, 2026", styles["title_center"]),
        PageBreak(),
    ]


def _table_from_markdown(
    lines: list[str], styles: dict[str, ParagraphStyle], available_width: float
) -> Table:
    parsed: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        parsed.append(cells)
    column_count = max(len(row) for row in parsed)
    data = []
    for row_index, row in enumerate(parsed):
        style = styles["table_header"] if row_index == 0 else styles["table_cell"]
        padded = row + [""] * (column_count - len(row))
        data.append([Paragraph(_inline_markup(cell), style) for cell in padded])
    widths = [available_width / column_count] * column_count
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7E7E7")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _image_flowable(relative_path: str, available_width: float) -> Image:
    image = Image(str(PROJECT_DIR / relative_path))
    max_width = available_width
    max_height = 170 * mm
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    return image


def _markdown_story(
    text: str, styles: dict[str, ParagraphStyle], available_width: float
) -> list[Any]:
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line == "## ВВЕДЕНИЕ")
    lines = lines[start:]
    story: list[Any] = []
    first_section = True
    index = 0
    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_buffer:
            combined = " ".join(part.strip().rstrip("  ") for part in paragraph_buffer)
            story.append(Paragraph(_inline_markup(combined), styles["body"]))
            paragraph_buffer.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            index += 1
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            title = stripped[3:]
            if not first_section:
                story.append(PageBreak())
            first_section = False
            structural = title in {
                "ВВЕДЕНИЕ",
                "ЗАКЛЮЧЕНИЕ",
                "СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ",
                "ПРИЛОЖЕНИЕ А",
            }
            story.append(
                Paragraph(
                    _inline_markup(title),
                    styles["structural"] if structural else styles["section"],
                )
            )
            index += 1
            continue

        if stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(_inline_markup(stripped[4:]), styles["subsection"]))
            index += 1
            continue

        if stripped == "$$":
            flush_paragraph()
            formula_lines: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != "$$":
                formula_lines.append(lines[index].strip())
                index += 1
            story.append(
                Paragraph(
                    html.escape(_latex_to_text(" ".join(formula_lines))),
                    styles["formula"],
                )
            )
            index += 1
            continue

        image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            flush_paragraph()
            story.append(Spacer(1, 2 * mm))
            story.append(_image_flowable(image_match.group(2), available_width))
            index += 1
            continue

        if stripped.startswith("Рисунок "):
            flush_paragraph()
            story.append(Paragraph(_inline_markup(stripped), styles["figure_caption"]))
            index += 1
            continue

        if stripped.startswith("Таблица "):
            flush_paragraph()
            story.append(Paragraph(_inline_markup(stripped), styles["table_caption"]))
            index += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            story.append(_table_from_markdown(table_lines, styles, available_width))
            story.append(Spacer(1, 3 * mm))
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            code_text = "<br/>".join(html.escape(code_line) for code_line in code_lines)
            story.append(Paragraph(code_text, styles["code"]))
            index += 1
            continue

        if re.match(r"^(?:- |\d+\. )", stripped):
            flush_paragraph()
            story.append(Paragraph(_inline_markup(stripped), styles["body"]))
            index += 1
            continue

        paragraph_buffer.append(stripped)
        index += 1

    flush_paragraph()
    return story


def _page_number(canvas: Any, document: Any) -> None:
    if document.page <= 1:
        return
    canvas.saveState()
    canvas.setFont("ReportSerif", 12)
    canvas.drawCentredString(A4[0] / 2, 10 * mm, str(document.page))
    canvas.restoreState()


def build_pdf() -> Path:
    """Собрать PDF и вернуть его путь."""

    font, bold_font, _italic_font = _register_fonts()
    styles = _styles(font, bold_font)
    document = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=A4,
        leftMargin=30 * mm,
        rightMargin=15 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="Лабораторная работа № 2 — Барышев А. А.",
        author="Барышев А. А.",
        subject="Методы оптимизации SciPy и Optuna",
    )
    report_text = REPORT_MD.read_text(encoding="utf-8")
    story = _title_page(styles)
    story.extend(_markdown_story(report_text, styles, document.width))
    document.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    return REPORT_PDF


if __name__ == "__main__":
    created = build_pdf()
    print(f"PDF создан: {created.name}")
