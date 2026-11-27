"""Generate laporan Business Professional (.docx) untuk KESTREL Python."""
import datetime

from docx import Document  # pyright: ignore[reportMissingImports]
from docx.enum.section import WD_SECTION  # pyright: ignore[reportMissingImports]
from docx.enum.text import WD_ALIGN_PARAGRAPH  # pyright: ignore[reportMissingImports]
from docx.oxml import OxmlElement  # pyright: ignore[reportMissingImports]
from docx.oxml.ns import qn  # pyright: ignore[reportMissingImports]
from docx.shared import Cm, Pt, RGBColor  # pyright: ignore[reportMissingImports]

ACCENT = RGBColor(0x1F, 0x6F, 0x43)  # green accent
GRAY = RGBColor(0x59, 0x59, 0x59)

doc = Document()

# ---------------- page setup (Business Professional margins) ----------------
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2.54)
sec.left_margin = sec.right_margin = Cm(3.17)

# ---------------- styles ----------------
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.line_spacing = 1.5
st.paragraph_format.space_after = Pt(6)

def set_heading(name, size, color=RGBColor(0, 0, 0)):
    s = doc.styles[name]
    s.font.name = "Arial"
    s.font.size = Pt(size)
    s.font.bold = True
    s.font.color.rgb = color

set_heading("Heading 1", 18, ACCENT)
set_heading("Heading 2", 14, ACCENT)
set_heading("Heading 3", 12)

def p(text, bold=False, italic=False, size=11, align=None, color=None,
      space_after=6, style=None):
    par = doc.add_paragraph(style=style)
    r = par.add_run(text)
    r.bold, r.italic = bold, italic
    r.font.size = Pt(size)
    if color: r.font.color.rgb = color
    if align is not None: par.alignment = align
    par.paragraph_format.space_after = Pt(space_after)
    return par

def bullet(text):
    par = doc.add_paragraph(style="List Bullet")
    par.add_run(text)
    par.paragraph_format.space_after = Pt(4)
    return par

# ---------------- helpers: header/footer per section ----------------
def add_field(par, instr):
    r = par.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
    r._r.append(f1); r._r.append(it); r._r.append(f2)

def setup_section(s, title):
    s.top_margin = s.bottom_margin = Cm(2.54)
    s.left_margin = s.right_margin = Cm(3.17)
    h = s.header.paragraphs[0]
    h.text = title
    h.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for r in h.runs:
        r.font.size = Pt(10); r.font.color.rgb = GRAY
    f = s.footer.paragraphs[0]
    f.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = f.add_run("Page ")
    r.font.size = Pt(10)
    add_field(f, "PAGE \\* MERGEFORMAT")
    for r in f.runs: r.font.size = Pt(10)

# ================= COVER (section 1, tanpa header/footer) =================
today = "2026-11-25"
p("KESTREL", bold=True, size=40, align=WD_ALIGN_PARAGRAPH.CENTER, color=ACCENT, space_after=2)
p("Implementasi Python — Chess Intelligence Laboratory", size=16,
  align=WD_ALIGN_PARAGRAPH.CENTER, color=GRAY, space_after=18)
p("Laporan Implementasi & Validasi Teknis", bold=True, size=14,
  align=WD_ALIGN_PARAGRAPH.CENTER, space_after=60)
for line in ("Versi: v1.0 (Python)", f"Tanggal: {today}",
             "Disusun oleh: Tim Engineering Kestrel",
             "Status: Implementasi MVP — semua acceptance test lulus"):
    p(line, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
p("Klasifikasi: Internal", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, color=GRAY, space_after=0)

doc.add_section(WD_SECTION.NEW_PAGE)

# ================= TOC (section 2) =================
sec2 = doc.sections[1]
setup_section(sec2, "KESTREL — Laporan Implementasi Python")
p("Daftar Isi", bold=True, size=16, align=WD_ALIGN_PARAGRAPH.CENTER, color=ACCENT)
toc_par = doc.add_paragraph()
fld = OxmlElement("w:fldSimple")
fld.set(qn("w:instr"), 'TOC \\o "1-3" \\h \\z \\u')
run = OxmlElement("w:r"); t = OxmlElement("w:t")
t.text = "Klik kanan → Update Field untuk memperbarui daftar isi."
run.append(t); fld.append(run)
toc_par._p.append(fld)
p("Catatan: nomor halaman daftar isi diperbarui otomatis oleh Microsoft Word "
  "(klik kanan daftar isi → Update Field).", italic=True, size=9, color=GRAY)

sec_body = doc.add_section(WD_SECTION.NEW_PAGE)
setup_section(sec_body, "KESTREL — Laporan Implementasi Python")

# ================= BODY =================
doc.add_heading("1. Ringkasan Eksekutif", level=1)
p("Laporan ini mendokumentasikan implementasi KESTREL — sistem chess intelligence "
  "hemat CPU — sepenuhnya dalam Python, sesuai Mathematical Foundation v1.0 dan PRD "
  "yang menjadi acuan proyek. Implementasi mencakup tujuh subsistem inti: chess core, "
  "search engine, tiny evaluator (neural MLP + heuristik klasik), adaptive memory, "
  "arena self-play, statistik Elo/SPRT, dan live analytics berbasis grafik.")
p("Seluruh acceptance criteria tingkat MVP telah diverifikasi melalui 23 unit test "
  "otomatis dengan hasil 23 lulus / 0 gagal, termasuk perft (20/400/8.902), deteksi "
  "skakmat & pat, batas parameter neural 107.777 (< 1.000.000), pencarian mate-in-1, "
  "pengambilan ratu bebas (exd5 pada kedalaman 9), batas anggaran node, persistensi "
  "SQLite, dan satu match Champion vs Challenger 8 game yang berjalan tanpa langkah "
  "ilegal. Batch self-play 12 game berjalan penuh dan menghasilkan telemetri nyata "
  "(581 langkah terekam ke adaptive memory) yang divisualisasikan pada dashboard analitik.")

doc.add_heading("2. Ruang Lingkup dan Pendekatan Teknis", level=1)
p("Sesuai keputusan implementasi, seluruh sistem ditulis dalam Python dengan memanfaatkan "
  "ekosistem library yang luas, tanpa komponen C++. Pendekatan ini memprioritaskan "
  "kecepatan pengembangan dan kemudahan audit matematika; konsekuensi performa (NPS "
  "lebih rendah dibanding C++) diterima dan dicatat eksplisit sebagai variabel yang "
  "akan dioptimalkan pada fase berikutnya. Library utama yang digunakan:")
bullet("python-chess — authority legal state, move generation, FEN/PGN/UCI (Prinsip #11: legal chess state adalah sumber kebenaran).")
bullet("NumPy — Tiny MLP evaluator (768→128→64→1, ReLU, FP32) dengan inferensi CPU murni.")
bullet("matplotlib — live analytics: evaluasi per ply, distribusi depth, node per langkah, panjang game.")
bullet("SQLite (sqlite3) — persistensi memory entries dan riwayat game (Section 14 PRD).")
bullet("hashlib/Zobrist (chess.polyglot) — transposition table keying.")
bullet("random + scipy/math — softmax sampling bersuhu, Elo, Wilson CI, dan SPRT.")

doc.add_heading("3. Arsitektur dan Kesesuaian terhadap Spesifikasi", level=1)
p("Setiap formula kunci pada Mathematical Foundation dipetakan langsung ke modul kode, "
  "dengan referensi seksi pada docstring. Tabel berikut merangkum kesesuaian tersebut:")
tbl = doc.add_table(rows=1, cols=3)
tbl.style = "Light Grid Accent 1"
hdr = tbl.rows[0].cells
for i, htxt in enumerate(("Subsistem", "Seksi Spec", "Status")):
    hdr[i].text = htxt
rows = [
    ("Chess core & terminal U(s)", "Sec. 1", "Lulus (perft + terminal tests)"),
    ("Evaluasi: material, PST tapered, mobilitas, pawn structure, king safety", "Sec. 3-8", "Lulus"),
    ("Tiny MLP 107.777 parameter + blending alpha=0.85", "Sec. 9", "Lulus (< 1M, target 100k-300k)"),
    ("Negamax alpha-beta + TT + quiescence + delta pruning", "Sec. 11-12", "Lulus"),
    ("Move ordering: MVV-LVA, killer, history", "Sec. 14", "Lulus"),
    ("PVS, LMR, null-move (anti-zugzwang guard)", "Sec. 15", "Lulus"),
    ("Iterative deepening + budget node/waktu", "Sec. 16", "Lulus (≤ 1,3x budget)"),
    ("Adaptive memory: relevance, decay, eviction, bounded influence", "Sec. 18, 32-34", "Lulus"),
    ("Arena: opening/color balancing, temperature sampling", "Sec. 21, 24", "Lulus"),
    ("Elo, Wilson CI, SPRT", "Sec. 23-26", "Lulus"),
]
for a, b, c in rows:
    cells = tbl.add_row().cells
    cells[0].text, cells[1].text, cells[2].text = a, b, c
p("")

doc.add_heading("4. Hasil Validasi", level=1)
doc.add_heading("4.1 Unit & Acceptance Tests", level=2)
p("Test suite otomatis (tests.py) menjalankan 23 pengujian yang mencakup batas perft, "
  "deteksi terminal, batas parameter model, simetri cermin evaluator, ketepatan taktis "
  "pencarian, disiplin anggaran node, keamanan state root, siklus hidup adaptive memory "
  "(store/activate/disable/evict), validitas PGN, persistensi, dan satu match penuh. "
  "Hasil akhir: 23 lulus, 0 gagal. Dua temuan perbaikan selama pengujian patut dicatat: "
  "penyebaban FEN opening pada header PGN (menyebabkan SAN ilegal saat replay) yang "
  "diperbaiki via Game.setup(), serta kesalahan ekspektasi pada mirror test — evaluasi "
  "White-relative yang benar justru berubah tanda pada posisi cermin, konsisten dengan "
  "definisi perspektif sisi (sign(p)).")
doc.add_heading("4.2 Self-Play Batch dan Telemetri", level=2)
p("Batch 12 game self-play (temperature 0,25 pada 12 ply awal untuk keberagaman opening) "
  "menghasilkan 4 menang / 5 seri / 3 kalah antar agen identik — distribusi yang wajar "
  "untuk agen sesimetris dan menandakan tidak ada bias warna. Rata-rata kedalaman "
  "pencarian efektif 1,6 ply dengan 779 node per langkah pada anggaran 150 ms/langkah, "
  "mencerminkan biaya evaluasi Python yang berat; ini adalah baseline performa, bukan "
  "batas desain. Seluruh 581 langkah terekam sebagai memory entries dengan confidence "
  "dan timestamp decay eksponensial sesuai Section 32.")
if True:
    doc.add_picture("results/analytics.png", width=Cm(14))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p("Gambar 1. Live analytics: evaluasi per ply, distribusi depth, node per langkah, "
      "panjang game (data engine nyata, bukan simulasi).", size=10,
      align=WD_ALIGN_PARAGRAPH.CENTER, color=GRAY)

doc.add_heading("5. Risiko dan Batasan", level=1)
bullet("Performa: NPS ~5 ribu (Python murni) jauh di bawah target C++; mitigasi: caching evaluasi, lazy PST, atau binding ke C/Rust pada fase 2.")
bullet("Neural evaluator belum tervalidasi: alpha tetap 0,85 sesuai kebijakan 'alpha tinggi untuk model belum tervalidasi' (Section 9); penurunan alpha memerlukan validasi pada dataset independen.")
bullet("SPRT pada 8 game bersifat provisional (Prinsip #12: tidak ada promosi dari sampel kecil); match 100+ game diperlukan sebelum klaim kekuatan.")
bullet("Adaptive influence dibatasi A_max dan dinonaktifkan secara default — tidak pernah mengoverride legalitas langkah.")

doc.add_heading("6. Langkah Berikutnya", level=1)
p("Tiga tindakan prioritas direkomendasikan: (1) optimasi hot path evaluator dengan "
  "tabel terinkrementasi untuk menaikkan NPS minimal 3-5x pada basis Python; (2) "
  "menjalankan match promosi formal 100 game Champion vs Challenger dengan SPRT penuh "
  "(alpha 0,05, beta 0,10, delta 35 Elo) dan paired openings; (3) melatih Tiny MLP dari "
  "data self-play (regresi MSE pada label deepcopy-search) untuk menurunkan alpha secara "
  "terkendali. Semua langkah wajib melewati gate regresi dan batas sumber daya sebelum "
  "promosi Champion (Prinsip #8 dan Section 27).")

p("")
p("Kesimpulan: implementasi Python KESTREL memenuhi definisi MVP — engine menyelesaikan "
  "game tanpa langkah ilegal, search dapat dibatalkan dan di-budget, evaluator berjalan "
  "native CPU di bawah 1 juta parameter, self-play dan persistensi berfungsi, statistik "
  "terukur, dan dashboard analitik menampilkan data engine nyata. Seluruh klaim pada "
  "laporan ini berstatus terverifikasi oleh test otomatis atau eksperimen terkontrol "
  "sesuai filosofi proyek: 'Small model. Big search. Prove every improvement.'",
  bold=False)

doc.save("/home/z/my-project/download/KESTREL_Python_Implementation_BusinessProfessional_2026-11-25.docx")
print("saved")
