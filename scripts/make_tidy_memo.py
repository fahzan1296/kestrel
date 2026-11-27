"""Buat memo singkat pembenahan folder (Business Professional, Fast writing)."""
from docx import Document  # pyright: ignore[reportMissingImports]
from docx.enum.section import WD_SECTION  # pyright: ignore[reportMissingImports]
from docx.enum.text import WD_ALIGN_PARAGRAPH  # pyright: ignore[reportMissingImports]
from docx.oxml import OxmlElement  # pyright: ignore[reportMissingImports]
from docx.oxml.ns import qn  # pyright: ignore[reportMissingImports]
from docx.shared import Cm, Pt, RGBColor  # pyright: ignore[reportMissingImports]

ACCENT = RGBColor(0x1F, 0x6F, 0x43)
GRAY = RGBColor(0x59, 0x59, 0x59)
OUT = "/home/z/my-project/download/PEMBENAHAAN_FOLDER_KESTREL_BusinessProfessional_2026-11-25.docx"

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2.54)
sec.left_margin = sec.right_margin = Cm(3.17)

st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.line_spacing = 1.5
st.paragraph_format.space_after = Pt(6)

def p(text, bold=False, size=11, align=None, color=None, space_after=6, style=None):
    par = doc.add_paragraph(style=style)
    r = par.add_run(text); r.bold = bold; r.font.size = Pt(size)
    if color: r.font.color.rgb = color
    if align is not None: par.alignment = align
    par.paragraph_format.space_after = Pt(space_after)
    return par

def add_field(par, instr):
    r = par.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
    r._r.append(f1); r._r.append(it); r._r.append(f2)

def setup_section(s):
    s.top_margin = s.bottom_margin = Cm(2.54)
    s.left_margin = s.right_margin = Cm(3.17)
    h = s.header.paragraphs[0]; h.text = "Memo Pembenahan Folder — KESTREL Python"
    h.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for r in h.runs: r.font.size = Pt(10); r.font.color.rgb = GRAY
    f = s.footer.paragraphs[0]; f.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = f.add_run("Page "); r.font.size = Pt(10)
    add_field(f, "PAGE \\* MERGEFORMAT")
    for r in f.runs: r.font.size = Pt(10)

# ---------- Cover ----------
p("KESTREL — Python", bold=True, size=34, align=WD_ALIGN_PARAGRAPH.CENTER, color=ACCENT, space_after=2)
p("Memo Pembenahan Struktur Folder", size=15, align=WD_ALIGN_PARAGRAPH.CENTER, color=GRAY, space_after=50)
p("Business Professional | 2026-11-25", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
p("Status: Selesai — struktur dirapikan, 23/23 tests tetap PASS", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)

sec2 = doc.add_section(WD_SECTION.NEW_PAGE); setup_section(sec2)

# ---------- Body ----------
doc.add_heading("1. Ringkasan", level=1)
p("Folder proyek `kestrel_python/` telah dibenahkan dari struktur datar menjadi struktur "
  "terstandar berbasis konvensi package Python. Seluruh perpindahan file bersifat relokasi "
  "saja — tidak ada perubahan logika kode. Verifikasi pasca-pembenahan: test suite lengkap "
  "dijalankan ulang dengan hasil 23 lulus / 0 gagal.")

doc.add_heading("2. Struktur Baru", level=1)
tbl = doc.add_table(rows=1, cols=3); tbl.style = "Light Grid Accent 1"
hdr = tbl.rows[0].cells
for i, t in enumerate(("Lokasi", "Isi", "Owner")):
    hdr[i].text = t
rows = [
    ("kestrel/", "Package inti: core.py, eval.py, search.py, memory.py, arena.py", "Tim Engine"),
    ("tests/", "test_kestrel.py — 23 unit/acceptance tests", "QA"),
    ("scripts/", "selfplay_demo.py, make_report.py (tool eksekusi)", "Tim Engine"),
    ("docs/", "Dokumen pendukung (baru)", "PM"),
    ("results/", "analytics.png, games.pgn, kestrel.db — artefak validasi nyata", "QA"),
    ("README.md", "Diperbarui: peta struktur baru + cara menjalankan", "PM"),
]
for a, b, c in rows:
    cells = tbl.add_row().cells
    cells[0].text, cells[1].text, cells[2].text = a, b, c
p("")

doc.add_heading("3. Tindakan Teknis yang Dilakukan", level=1)
p("Tiga tindakan dilakukan: (1) relokasi test ke `tests/` dan skrip eksekusi ke `scripts/`; "
  "(2) perbaikan `sys.path` pada ketiga file terpindah agar dapat dijalankan dari direktori "
  "mana pun (path absolut berbasis lokasi file); (3) pembersihan `__pycache__`. File lama "
  "di root tinggal README.md dan folder — root kini bersih dan self-explanatory.")

doc.add_heading("4. Verifikasi & Langkah Berikutnya", level=1)
p("Pasca-relokasi, `python3 tests/test_kestrel.py` dijalankan dari root dan menghasilkan "
  "23/23 PASS — termasuk match Champion vs Challenger penuh tanpa langkah ilegal. "
  "Langkah berikutnya (Owner: Tim Engine, prioritas tinggi): menambahkan `pyproject.toml` "
  "agar package dapat di-install (`pip install -e .`), serta CI sederhana yang menjalankan "
  "test suite pada setiap perubahan.")

doc.save(OUT)
print("saved:", OUT)
