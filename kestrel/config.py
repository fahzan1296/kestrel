"""KESTREL 1.6 — Konfigurasi terpusat (satu sumber kebenaran, format TOML).

Semua komponen (trainer unified, selfplay, REDA, learning, verification,
knowledge, CLI, dashboard, lichess) membaca dari modul ini. Nilai default
didefinisikan sebagai dataclass; ``config.toml`` menimpa nilai default per
section; CLI menimpa lagi per flag. Urutan prioritas: default < TOML < CLI.

Contoh:

    from kestrel.config import load_config
    cfg = load_config()                      # config.toml di root proyek
    cfg = load_config("custom.toml")         # file lain
    cfg = cfg.with_overrides(games=40)       # override CLI/datetime
"""
from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _coerce(cls, data: dict):
    """Bangun dataclass `cls` dari dict TOML, buang key tak dikenal."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class SelfplayConfig:
    """Konfigurasi batch self-play — sumber utama budget untuk trainer."""
    games_per_batch: int = 100
    concurrency: int = 2
    time_ms: int = 150
    max_depth: int = 10
    min_depth: int = 6          # jaminan depth (grace time per langkah)
    grace_ms: int = 600
    max_plies: int = 160
    temperature: float = 0.25
    opening_diversity_plies: int = 12
    tt_size: int = 1 << 18
    use_book: bool = True       # langkah pembuka dari knowledge base


@dataclass
class SearchConfig:
    min_depth: int = 6
    grace_ms: int = 1000
    max_depth: int = 64
    move_time_ms: int = 1000
    node_limit: int = 0
    tt_size: int = 1 << 18
    use_null_move: bool = True
    use_lmr: bool = True
    use_lmp: bool = True
    check_extension: bool = True


@dataclass
class LearningConfig:
    epochs: int = 10
    lr: float = 0.015
    batch_size: int = 64
    patience: int = 4
    validation_fraction: float = 0.20
    seed: int = 42
    max_samples_per_game: int = 120


@dataclass
class RedaConfig:
    enabled: bool = True
    max_active_features: int = 3
    max_abs_residual_weight: float = 0.02
    learning_rate: float = 0.01
    l2: float = 0.002
    min_validation_improvement: float = 1e-5
    verification_depth: int = 12
    verification_time_ms: int = 1500
    max_candidate_positions_per_batch: int = 24
    max_tracked_positions: int = 100000
    verification_steps: int = 160
    auto_after_batch: bool = False


@dataclass
class EvalConfig:
    """Verifikasi otomatis champion vs challenger (tiap generasi)."""
    games: int = 30
    time_ms: int = 300
    max_depth: int = 10
    min_depth: int = 6
    grace_ms: int = 600
    temperature: float = 0.0
    sprt_elo0: float = 0.0
    sprt_elo1: float = 35.0
    sprt_alpha: float = 0.05
    sprt_beta: float = 0.10
    min_score_rate: float = 0.50   # ambang promote bila SPRT belum H1
    max_tactical_regression: int = 0  # akurasi taktik boleh turun maks N soal
    require_consistency: bool = True
    min_nps_ratio: float = 0.60    # NPS challenger >= rasio champion
    seed: int = 2024


@dataclass
class TrainConfig:
    """Loop training unified: selfplay -> knowledge -> learning -> REDA ->
    verification -> promote/rollback, diulang per generasi."""
    generations: int = 10
    auto_promote: bool = True
    rollback_on_reject: bool = True
    keep_last_checkpoints: int = 5
    report_every: bool = True


@dataclass
class KnowledgeConfig:
    enabled: bool = True
    book_max_plies: int = 10
    book_min_games: int = 2        # dukungan minimum sebelum langkah book dipakai
    book_max_moves_per_position: int = 3
    db_name: str = "knowledge.db"


@dataclass
class ArenaConfig:
    minimum_games: int = 100
    promotion_mode: str = "sprt"   # "sprt" | "score" | "manual"


@dataclass
class CrossConfig:
    """Pengaturan sistem cross-engine: Kestrel vs Stockfish (1.7).

    ``parallel``: jumlah game yang dimainkan bersamaan oleh proses terpisah
    (masing-masing dengan Stockfish subprocess sendiri). ``adaptive_skill``:
    Skill Level Stockfish dinaikkan/diturunkan otomatis mengikuti score_rate
    Kestrel pada jendela game terakhir agar korpus tetap kompetitif.
    """
    games_per_batch: int = 20
    skill: int = 5
    skill_min: int = 0
    skill_max: int = 12
    adaptive_skill: bool = True
    skill_up_score: float = 0.62     # score_rate jendela >= ini -> skill +1
    skill_down_score: float = 0.32   # score_rate jendela <= ini -> skill -1
    skill_window: int = 4            # game terakhir yang dirata-rata
    parallel: int = 1
    stockfish_time_ms: int = 200
    time_ms: int = 150
    max_depth: int = 10
    max_plies: int = 160
    min_depth: int = 6
    grace_ms: int = 600
    use_book: bool = True


@dataclass
class LichessConfig:
    token: str = ""
    username: str = ""
    accept_bot: bool = True
    # Default TRUE: akun BOT di lichess memang dimainkan manusia.
    # (1.6 awal default False membuat bot menolak tantangan diam-diam.)
    accept_human: bool = True
    max_rating_gap: int = 400
    min_rating: int = 0
    concurrency: int = 1
    move_time_ms: int = 1000
    max_depth: int = 18
    min_depth: int = 10
    grace_ms: int = 1500
    book_moves: bool = True
    resign_cp: int = -900
    resign_moves: int = 3
    offer_draw_cp: int = 0
    poll_seconds: float = 2.0
    max_games: int = 0             # 0 = tanpa batas
    pgn_dir: str = "lichess"
    auto_seek: bool = True
    seek_interval_seconds: float = 60.0
    seek_pool_size: int = 100      # 1..512, sesuai GET /api/bot/online
    seek_cooldown_games: int = 10
    seek_rated: bool = False
    seek_clock_seconds: int = 300
    seek_increment_seconds: int = 3
    seek_color: str = "random"
    learn_from_games: bool = True
    learn_every_games: int = 5     # 0 = nonaktif
    learn_min_games: int = 8
    learn_max_games: int = 200


@dataclass
class StorageConfig:
    results_dir: str = "results"
    database: str = "results/kestrel.db"
    pgn_file: str = "results/games.pgn"


@dataclass
class AppConfig:
    selfplay: SelfplayConfig = field(default_factory=SelfplayConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    reda: RedaConfig = field(default_factory=RedaConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    cross: CrossConfig = field(default_factory=CrossConfig)
    lichess: LichessConfig = field(default_factory=LichessConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    config_path: str = str(ROOT / "config.toml")

    # ---------------- derived paths ----------------
    @property
    def results_path(self) -> Path:
        p = Path(self.storage.results_dir)
        return p if p.is_absolute() else ROOT / p

    @property
    def pgn_path(self) -> Path:
        p = Path(self.storage.pgn_file)
        return p if p.is_absolute() else ROOT / p

    @property
    def db_path(self) -> Path:
        p = Path(self.storage.database)
        return p if p.is_absolute() else ROOT / p

    @property
    def knowledge_db_path(self) -> Path:
        return self.results_path / self.knowledge.db_name

    @property
    def champion_model_path(self) -> Path:
        return self.results_path / "learned_eval.json"

    @property
    def champion_meta_path(self) -> Path:
        return self.results_path / "champion.json"

    # ---------------- overrides ----------------
    def with_overrides(self, section: str, **kwargs) -> AppConfig:
        """Salin config dengan nilai bagian tertentu ditimpa (CLI override)."""
        cfg = dataclasses.replace(self)
        current = getattr(cfg, section)
        allowed = {f.name for f in fields(type(current))}
        cleaned = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        setattr(cfg, section, dataclasses.replace(current, **cleaned))
        return cfg

    def to_dict(self) -> dict:
        out = asdict(self)
        out["config_path"] = self.config_path
        return out


def load_config(path: str | Path | None = None) -> AppConfig:
    """Muat config.toml (atau file lain); section tak dikenal diabaikan."""
    target = Path(path) if path else ROOT / "config.toml"
    cfg = AppConfig(config_path=str(target))
    if target.exists():
        try:
            with target.open("rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"config.toml tidak valid: {target}") from exc
        mapping = {
            "selfplay": SelfplayConfig, "search": SearchConfig,
            "learning": LearningConfig, "reda": RedaConfig,
            "eval": EvalConfig, "train": TrainConfig,
            "knowledge": KnowledgeConfig, "arena": ArenaConfig,
            "cross": CrossConfig,
            "lichess": LichessConfig, "storage": StorageConfig,
        }
        for section, cls in mapping.items():
            if section in data and isinstance(data[section], dict):
                setattr(cfg, section, _coerce(cls, data[section]))
        # 1.5 kompatibilitas: [engine] lama memetakan ke [search]
        if "engine" in data and isinstance(data["engine"], dict):
            eng = data["engine"]
            patch = {}
            if "move_time_ms" in eng:
                patch["move_time_ms"] = eng["move_time_ms"]
            if "max_depth" in eng:
                patch["max_depth"] = eng["max_depth"]
            if "node_limit" in eng:
                patch["node_limit"] = eng["node_limit"]
            if patch:
                cfg.search = dataclasses.replace(cfg.search, **patch)
    return cfg


def validate_config(cfg: AppConfig) -> list[str]:
    """Validasi nilai; kembalikan daftar pesan error (kosong = valid)."""
    errors = []
    checks = [
        (cfg.selfplay.games_per_batch >= 1, "selfplay.games_per_batch >= 1"),
        (cfg.selfplay.concurrency >= 1, "selfplay.concurrency >= 1"),
        (cfg.selfplay.time_ms >= 10, "selfplay.time_ms >= 10"),
        (cfg.selfplay.max_depth >= 1, "selfplay.max_depth >= 1"),
        (0 <= cfg.selfplay.min_depth <= cfg.selfplay.max_depth,
         "selfplay.min_depth <= max_depth"),
        (cfg.eval.games >= 2 and cfg.eval.games % 2 == 0,
         "eval.games harus genap dan >= 2 (warna seimbang)"),
        (0 <= cfg.eval.min_depth <= cfg.eval.max_depth,
         "eval.min_depth <= eval.max_depth"),
        (cfg.train.generations >= 1, "train.generations >= 1"),
        (1 <= cfg.knowledge.book_max_plies <= 40, "knowledge.book_max_plies 1..40"),
        (cfg.lichess.move_time_ms >= 100, "lichess.move_time_ms >= 100"),
        (0 <= cfg.lichess.min_depth <= cfg.lichess.max_depth,
         "lichess.min_depth <= max_depth"),
        (cfg.cross.games_per_batch >= 1, "cross.games_per_batch >= 1"),
        (0 <= cfg.cross.skill_min <= cfg.cross.skill <= cfg.cross.skill_max <= 20,
         "cross.skill_min <= skill <= skill_max <= 20"),
        (cfg.cross.parallel >= 1, "cross.parallel >= 1"),
        (cfg.cross.stockfish_time_ms >= 10, "cross.stockfish_time_ms >= 10"),
        (cfg.cross.time_ms >= 10, "cross.time_ms >= 10"),
        (cfg.cross.max_depth >= 1, "cross.max_depth >= 1"),
        (0 <= cfg.cross.min_depth <= cfg.cross.max_depth,
         "cross.min_depth <= cross.max_depth"),
        (0.0 <= cfg.cross.skill_down_score < cfg.cross.skill_up_score <= 1.0,
         "cross.skill_down_score < skill_up_score <= 1"),
        (cfg.cross.skill_window >= 1, "cross.skill_window >= 1"),
    ]
    for ok, msg in checks:
        if not ok:
            errors.append(msg)
    return errors
