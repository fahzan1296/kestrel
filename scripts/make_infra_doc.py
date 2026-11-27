"""Memo infrastruktur & pipeline KESTREL — Business Professional, Fast writing."""
from docx import Document  # pyright: ignore[reportMissingImports]
from docx.enum.section import WD_SECTION  # pyright: ignore[reportMissingImports]
from docx.enum.text import WD_ALIGN_PARAGRAPH  # pyright: ignore[reportMissingImports]
from docx.oxml import OxmlElement  # pyright: ignore[reportMissingImports]
from docx.oxml.ns import qn  # pyright: ignore[reportMissingImports]
from docx.shared import Cm, Pt, RGBColor  # pyright: ignore[reportMissingImports]

ACCENT = RGBColor(0x1F, 0x6F, 0x43); GRAY = RGBColor(0x59, 0x59, 0x59)
OUT = "/home/z/my-project/download/KESTREL_Infrastruktur_Pipeline_BusinessProfessional_2026-11-25.docx"
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2.54)
sec.left_margin = sec.right_margin = Cm(3.17)
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.line_spacing = 1.5; st.paragraph_format.space_after = Pt(6)

def p(text, bold=False, size=11, align=None, color=None, style=None, space_after=6):
    par = doc.add_paragraph(style=style); r = par.add_run(text)
    r.bold = bold; r.font.size = Pt(size)
    if color: r.font.color.rgb = color
    if align is not None: par.alignment = align
    par.paragraph_format.space_after = Pt(space_after); return par

def add_field(par, instr):
    r = par.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
    r._r.append(f1); r._r.append(it); r._r.append(f2)

def setup(s):
    s.top_margin = s.bottom_margin = Cm(2.54); s.left_margin = s.right_margin = Cm(3.17)
    h = s.header.paragraphs[0]; h.text = "KESTREL — Infrastruktur, Pipeline & Benchmark"
    h.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for r in h.runs: r.font.size = Pt(10); r.font.color.rgb = GRAY
    f = s.footer.paragraphs[0]; f.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = f.add_run("Page "); r.font.size = Pt(10); add_field(f, "PAGE \\* MERGEFORMAT")
    for r in f.runs: r.font.size = Pt(10)

p("KESTREL — Python", bold=True, size=34, align=WD_ALIGN_PARAGRAPH.CENTER, color=ACCENT, space_after=2)
p("Infrastruktur, Pipeline & Benchmark", size=15, align=WD_ALIGN_PARAGRAPH.CENTER, color=GRAY, space_after=50)
p("Business Professional | 2026-11-25", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
p("Status: Pipeline quick-mode PASS | Paket: kestrel_python.zip", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)

s2 = doc.add_section(WD_SECTION.NEW_PAGE); setup(s2)

p("Daftar Isi", bold=True, size=14, color=ACCENT)
toc = doc.add_paragraph()
fld = OxmlElement("w:fldSimple"); fld.set(qn("w:instr"), 'TOC \\o "1-2" \\h \\z \\u')
run = OxmlElement("w:r"); t = OxmlElement("w:t")
t.text = "Klik kanan -> Update Field."; run.append(t); fld.append(run)
toc._p.append(fld)

doc.add_section(WD_SECTION.NEW_PAGE); setup(doc.sections[-1])

doc.add_heading("1. Ringkasan", level=1)
p("Proyek KESTREL Python kini dilengkapi infrastruktur produksi: pipeline end-to-end "
  "satu perintah, benchmark suite, integrasi Stockfish via UCI, CI GitHub Actions, "
  "packaging (pyproject.toml), Docker, dan konfigurasi TOML sesuai Section 16 PRD. "
  "Pipeline quick-mode telah dijalankan dengan hasil keseluruhan PASS.")

doc.add_heading("2. Pipeline End-to-End", level=1)
p("Perintah `python3 -m kestrel.pipeline` menjalankan lima tahap berurutan dengan "
  "laporan JSON otomatis per eksekusi: (1) tests — 23 unit/acceptance tests; "
  "(2) self-play batch + live analytics; (3) benchmark — tactical suite, performa NPS, "
  "konsistensi evaluasi; (4) match Champion vs Challenger balanced; (5) sparring "
  "Stockfish (opsional, skip otomatis bila binary tidak ada). Mode `--quick` tersedia "
  "untuk smoke test CI, dan setiap tahap dapat di-skip individual.")
tbl = doc.add_table(rows=1, cols=3); tbl.style = "Light Grid Accent 1"
hdr = tbl.rows[0].cells
for i, t in enumerate(("Tahap", "Output", "Owner")): hdr[i].text = t
for a, b, c in [
    ("tests", "23/23 PASS (gate rilis)", "QA"),
    ("selfplay", "games.pgn, kestrel.db, analytics.png", "Tim Engine"),
    ("benchmark", "tactical accuracy, NPS, drift check", "QA"),
    ("match", "score rate + Elo delta + SPRT", "Tim Engine"),
    ("stockfish", "skor sparring (provisional < 100 game)", "Tim Engine"),
]:
    cells = tbl.add_row().cells
    cells[0].text, cells[1].text, cells[2].text = a, b, c
p("")

doc.add_heading("3. Benchmark & Stockfish", level=1)
p("Benchmark suite (Section 29 spec) menyediakan tactical suite 5 posisi (mate-in-1, "
  "queen grab, back-rank, castling) dengan metrik TacticalAccuracy, performance suite "
  "pada posisi kiwipete untuk NPS, dan evaluation consistency check. Wrapper Stockfish "
  "(Section 10 PRD) mengelola proses UCI dengan skill level/hash/threads configurable, "
  "graceful fallback bila binary tidak tersedia, dan penandaan provisional untuk "
  "sampel kecil sesuai kebijakan promosi.")

doc.add_heading("4. Packaging, CI & Deployment", level=1)
p("`pyproject.toml` membuat proyek ter-install (`pip install -e .`); `config.toml` "
  "memusatkan seluruh parameter sesuai Section 16 PRD; `Makefile` menyediakan target "
  "install/test/pipeline/quick/zip. GitHub Actions menjalankan test + pipeline quick "
  "pada setiap push dan mengunggah artefak results; Dockerfile membundel engine dengan "
  "Stockfish pre-installed untuk lingkungan reproduksibel (Section 38: reproducibility).")

doc.add_heading("5. Kesimpulan & Langkah Berikutnya", level=1)
p("Infrastruktur menutup celah operasional MVP: validasi, pengukuran, kompetisi, dan "
  "distribusi kini satu perintah. Langkah berikutnya (Owner: Tim Engine): aktifkan CI "
  "di repositori, jalankan pipeline full-mode malam hari, dan pasang binary Stockfish "
  "di runner untuk mengaktifkan tahap sparring. Arsip terbaru: kestrel_python.zip.")

doc.save(OUT); print("saved:", OUT)
