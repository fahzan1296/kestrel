"""KESTREL 1.6 — CLI: satu pintu untuk semua operasi, tanpa konfigurasi UI.

Semua perintah membaca ``config.toml`` sebagai default lalu menimpanya dengan
flag baris-perintah — jadi eksperimen tidak lagi 1x per 1 konfigurasi UI.

Contoh cepat:

    python -m kestrel.cli train --generations 5 --games 40 --time-ms 200
    python -m kestrel.cli cross-train --games 20 --time-ms 150 --skill 5
    python -m kestrel.cli train --quick                 # smoke test ~2 menit
    python -m kestrel.cli eval --games 60               # eval champion vs kandidat
    python -m kestrel.cli selfplay --games 100 --concurrency 4
    python -m kestrel.cli book build                    # bangun book dari korpus
    python -m kestrel.cli verify                        # bukti kekuatan + konsistensi
    python -m kestrel.cli lichess --token XXXX          # jadi bot Lichess
    python -m kestrel.cli model history                 # riwayat champion
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import AppConfig, load_config, validate_config
from .versioning import CheckpointStore


def _add_common_overrides(p: argparse.ArgumentParser):
    p.add_argument("--config", help="path config.toml alternatif")
    p.add_argument("--games", type=int, help="override selfplay.games_per_batch")
    p.add_argument("--concurrency", type=int, help="override selfplay.concurrency")
    p.add_argument("--time-ms", type=int, help="override selfplay.time_ms")
    p.add_argument("--depth", type=int, help="override selfplay.max_depth")
    p.add_argument("--min-depth", type=int, help="override selfplay.min_depth (jaminan depth)")
    p.add_argument("--grace-ms", type=int, help="override selfplay.grace_ms")


def _apply_common(cfg: AppConfig, args) -> AppConfig:
    if args.games is not None:
        cfg = cfg.with_overrides("selfplay", games_per_batch=args.games)
    if args.concurrency is not None:
        cfg = cfg.with_overrides("selfplay", concurrency=args.concurrency)
    if args.time_ms is not None:
        cfg = cfg.with_overrides("selfplay", time_ms=args.time_ms)
    if args.depth is not None:
        cfg = cfg.with_overrides("selfplay", max_depth=args.depth)
    if args.min_depth is not None:
        cfg = cfg.with_overrides("selfplay", min_depth=args.min_depth)
    if args.grace_ms is not None:
        cfg = cfg.with_overrides("selfplay", grace_ms=args.grace_ms)
    return cfg


def _load_or_exit(args) -> AppConfig:
    cfg = load_config(args.config)
    errors = validate_config(cfg)
    if errors:
        print(json.dumps({"error": "konfigurasi tidak valid", "messages": errors},
                         indent=2))
        sys.exit(2)
    return cfg


# ---------------- command implementations ----------------

def cmd_config(args) -> int:
    cfg = load_config(args.config)
    if args.action == "path":
        print(cfg.config_path)
        return 0
    if args.action == "validate":
        errors = validate_config(cfg)
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return 0 if not errors else 2
    if args.action == "init":
        target = Path(args.write or "config.toml")
        default = (Path(__file__).resolve().parent.parent / "config.toml")
        target.write_text(default.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"config template ditulis ke {target}")
        return 0
    # show (default)
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
    return 0


def cmd_train(args) -> int:
    from .train_status import finish, start, update  # pyright: ignore[reportMissingImports]
    from .trainer import run_training
    cfg = _load_or_exit(args)
    cfg = _apply_common(cfg, args)
    if args.generations is not None:
        cfg = cfg.with_overrides("train", generations=args.generations)
    if args.quick:
        from .trainer import QuickOverrides
        cfg = QuickOverrides.apply(cfg)
    verify = not args.no_verify
    if args.eval_games is not None:
        cfg = cfg.with_overrides("eval", games=args.eval_games)
    if args.eval_time_ms is not None:
        cfg = cfg.with_overrides("eval", time_ms=args.eval_time_ms)
    settings = {
        "generations": cfg.train.generations,
        "games": cfg.selfplay.games_per_batch,
        "quick": args.quick,
        "verify": verify,
        "eval_games": cfg.eval.games,
        "eval_time_ms": cfg.eval.time_ms,
        "results_path": str(cfg.results_path),
    }
    run = start(cfg.results_path, entry_point="cli", settings=settings)
    rid = run["run_id"]

    def progress(info):
        detail = dict(info)
        message = str(detail.pop("message", "training"))
        prog = {k: detail[k] for k in ("stage", "generation", "completed",
                                       "total", "result") if k in detail}
        update(cfg.results_path, run_id=rid, message=message,
               progress=prog, **detail)

    try:
        rep = run_training(cfg, quick=args.quick, verify=verify, progress=progress)
    except Exception as exc:
        finish(cfg.results_path, run_id=rid, status="failed",
               error=f"{type(exc).__name__}: {exc}")
        raise
    result = {k: rep.get(k) for k in
              ("overall", "promoted_count", "final_generation", "report_path")}
    finish(cfg.results_path, run_id=rid,
           status=str(rep.get("overall", "FAIL")).lower(),
           result=result)
    print(json.dumps(result, indent=2))
    return 0 if rep.get("overall") == "PASS" else 1


def cmd_cross_train(args) -> int:
    """Cross-engine training: Kestrel vs Stockfish, lalu latih dari PGN itu."""
    from .cross_engine import run_cross_engine_training  # pyright: ignore[reportMissingImports]
    from .train_status import start  # pyright: ignore[reportMissingImports]

    cfg = _load_or_exit(args)
    run = start(cfg.results_path, entry_point="cli", settings={
        "mode": "cross_engine", "games": args.games or cfg.cross.games_per_batch,
        "time_ms": args.time_ms or cfg.cross.time_ms, "skill": args.skill,
        "parallel": args.parallel or cfg.cross.parallel,
        "adaptive_skill": not args.no_adaptive,
        "stockfish": args.stockfish or "auto",
        "generations": args.generations or cfg.train.generations,
        "quick": args.quick, "results_path": str(cfg.results_path)})
    result = run_cross_engine_training(
        cfg, run_id=run["run_id"], skill=args.skill,
        stockfish_path=args.stockfish, stockfish_time_ms=args.stockfish_time_ms,
        games=args.games, time_ms=args.time_ms, max_depth=args.depth,  # pyright: ignore[reportCallIssue]
        min_depth=args.min_depth, grace_ms=args.grace_ms,
        generations=args.generations, parallel=args.parallel,
        adaptive_skill=not args.no_adaptive, quick=args.quick,
        verify=not args.no_verify)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("overall") == "PASS" else 1

def cmd_selfplay(args) -> int:
    from .selfplay import run_selfplay
    cfg = _load_or_exit(args)
    cfg = _apply_common(cfg, args)
    store = CheckpointStore(cfg.results_path)
    kb = _open_knowledge(cfg) if cfg.selfplay.use_book else None
    rep = run_selfplay(
        games=cfg.selfplay.games_per_batch, concurrency=cfg.selfplay.concurrency,
        time_ms=cfg.selfplay.time_ms, max_depth=cfg.selfplay.max_depth,
        max_plies=cfg.selfplay.max_plies, temperature=cfg.selfplay.temperature,
        min_depth=cfg.selfplay.min_depth, grace_ms=cfg.selfplay.grace_ms,
        net_path=str(store.champion_model_path) if store.champion_model_path.exists() else None,
        book_path=str(kb.db_path) if kb else None,
        pgn_path=cfg.pgn_path, db_path=cfg.db_path)
    if args.learn:
        kb = kb or _open_knowledge(cfg)
        kb.learn_from_pgn([cfg.pgn_path])
        rep["knowledge_stats"] = kb.stats()
    print(json.dumps(rep, indent=2))
    return 0 if rep["games_finished"] > 0 else 1


def _open_knowledge(cfg: AppConfig):
    from .knowledge import KnowledgeBase
    return KnowledgeBase(cfg.knowledge_db_path, cfg.knowledge.book_max_plies,
                         cfg.knowledge.book_min_games)


def cmd_nnue(args) -> int:
    """Distilasi satu-tembak TinyNNUE dari korpus PGN, atau info net aktif."""
    cfg = _load_or_exit(args)
    store = CheckpointStore(cfg.results_path)
    if args.action == "info":
        net = store.load_champion()
        print(json.dumps({"kind": net.kind, "params": net.params(),
                          "champion_file": str(store.champion_model_path),
                          "champion_exists": store.has_trained_champion(),
                          "metadata": net.metadata}, indent=2, default=str))
        return 0
    paths = [str(p_) for p_ in (args.pgn or [str(cfg.pgn_path)])]
    missing = [p_ for p_ in paths if not Path(p_).exists()]
    if missing:
        print(f"PGN tidak ditemukan: {missing}")
        return 1
    from .nnue_train import train_from_pgn
    base = None
    if args.base_model:
        from .nnue import TinyNNUE
        base = TinyNNUE.load(args.base_model)
    elif store.has_trained_champion():
        base = store.load_champion()
    kwargs = {}
    if args.epochs is not None:
        kwargs["epochs"] = args.epochs
    if args.lr is not None:
        kwargs["lr"] = args.lr
    net, report = train_from_pgn(paths, base=base, **kwargs)
    out = Path(args.output) if args.output else store.champion_model_path
    net.save(out, {"trained_by": "cli nnue distill"})
    print(json.dumps({"output": str(out), "params": net.params(),
                      "samples": report["samples"],
                      "best_epoch": report["best_epoch"],
                      "validation_mse_cp2": report["best_validation_mse_cp2"]},
                     indent=2))
    return 0


def cmd_eval(args) -> int:
    """Head-to-head terperinci dua model (default: champion vs kandidat terakhir)."""
    from .nnue import TinyNNUE
    from .verification import report_markdown, verify_challenger
    cfg = _load_or_exit(args)
    store = CheckpointStore(cfg.results_path)
    if args.eval_games:
        cfg = cfg.with_overrides("eval", games=args.eval_games)
    if args.eval_time_ms:
        cfg = cfg.with_overrides("eval", time_ms=args.eval_time_ms)
    champ_path = args.champion or (str(store.champion_model_path)
                                   if store.champion_model_path.exists() else None)
    cand_path = args.challenger
    if cand_path is None:
        cands = store.candidates()
        cand_path = str(cands[-1]) if cands else None
    champ_model = TinyNNUE.load(champ_path) if champ_path else None
    cand_model = TinyNNUE.load(cand_path) if cand_path else None
    print(f"champion : {champ_path or '(baseline bawaan TinyNNUE)'}")
    print(f"challenger: {cand_path or '(baseline bawaan TinyNNUE)'}")
    rep = verify_challenger(champ_model, cand_model, cfg, verbose=not args.quiet)
    tag = f"{int(time.time())}"
    out = cfg.results_path / f"eval_manual_{tag}.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    (cfg.results_path / f"eval_manual_{tag}.md").write_text(report_markdown(rep), encoding="utf-8")
    print(json.dumps({"decision": rep["decision"], "score_rate": rep["match"]["score_rate"],
                      "elo_delta": rep["match"]["elo_delta"], "sprt": rep["match"]["sprt"],
                      "report": str(out)}, indent=2))
    return 0 if rep["decision"] == "PROMOTE" else 1


def cmd_verify(args) -> int:
    """Suite bukti kekuatan & konsistensi: benchmark penuh + match vs baseline.

    Champion aktif (atau kandidat --challenger) diadu vs net baseline bawaan
    paket (checkpoint terlatih), ditambah akurasi taktik, NPS, konsistensi.
    """
    from .benchmark import evaluation_consistency, performance_suite, tactical_suite
    from .nnue import TinyNNUE
    from .verification import report_markdown, verify_challenger
    cfg = _load_or_exit(args)
    store = CheckpointStore(cfg.results_path)
    model = None
    model_path = args.challenger or (str(store.champion_model_path)
                                     if store.champion_model_path.exists() else None)
    if model_path:
        model = TinyNNUE.load(model_path)
    rep = verify_challenger(None, model, cfg, quick=args.quick, verbose=not args.quiet)
    out = cfg.results_path / f"verify_{int(time.time())}.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    (cfg.results_path / f"verify_{int(time.time())}.md").write_text(report_markdown(rep), encoding="utf-8")
    if not args.quick:
        bench = {"tactical": tactical_suite(time_ms=2000),
                 "performance": performance_suite(node_budget=30000),
                 "consistency": evaluation_consistency(samples=120)}
        out2 = cfg.results_path / f"bench_{int(time.time())}.json"
        out2.write_text(json.dumps(bench, indent=2), encoding="utf-8")
        rep["full_benchmark"] = {"report": str(out2),
                                 "tactical_accuracy": bench["tactical"]["accuracy"]}
        out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"decision": rep["decision"], "vs": "baseline bawaan TinyNNUE",
                      "model": model_path or "(baseline)",
                      "score_rate": rep["match"]["score_rate"],
                      "elo_delta": rep["match"]["elo_delta"], "sprt": rep["match"]["sprt"],
                      "tactical": rep["challenger_bench"]["tactical_solved"],
                      "report": str(out)}, indent=2))
    return 0 if rep["decision"] == "PROMOTE" else 1


def cmd_bench(args) -> int:
    from .benchmark import (
        evaluation_consistency,
        performance_suite,
        special_move_suite,
        tactical_suite,
    )
    t = tactical_suite(time_ms=800 if args.quick else 3000,
                       node_limit=30000 if args.quick else 200000)
    s = special_move_suite()
    p = performance_suite(node_budget=10000 if args.quick else 50000)
    c = evaluation_consistency(samples=40 if args.quick else 200)
    ok = t["accuracy"] >= 0.6 and s["pass"] and c["within_epsilon"]
    out = {"tactical": {k: t[k] for k in ("solved", "total", "accuracy")},
           "special_moves": s["pass"], "nps": p["nps"],
           "consistency_drift": c["max_drift"], "pass": ok}
    if args.workers and args.workers > 1:
        from .benchmark import parallel_performance_suite
        pp = parallel_performance_suite(
            node_budget=10000 if args.quick else 50000, workers=args.workers)
        out["aggregate_nps"] = pp["aggregate_nps"]
        out["workers"] = pp["workers"]
        out["scaling_efficiency"] = pp["scaling_efficiency"]
    print(json.dumps(out, indent=2))
    return 0 if ok else 1


def cmd_book(args) -> int:
    cfg = _load_or_exit(args)
    kb = _open_knowledge(cfg)
    if args.action == "build":
        paths = [cfg.pgn_path] + [Path(p) for p in (args.pgn or [])]
        rep = kb.learn_from_pgn(paths, max_plies=args.max_plies)
        rep["stats"] = kb.stats()
        print(json.dumps(rep, indent=2))
        return 0
    if args.action == "show":
        print(json.dumps({"stats": kb.stats(),
                          "top_lines": kb.top_book_lines(args.limit),
                          "recent_lessons": kb.lessons(limit=args.limit)}, indent=2))
        return 0
    if args.action == "clear":
        kb.clear_book()
        print("book dibersihkan")
        return 0
    return 2


def cmd_model(args) -> int:
    cfg = _load_or_exit(args)
    store = CheckpointStore(cfg.results_path, keep_last=cfg.train.keep_last_checkpoints)
    if args.action == "show":
        net = store.load_champion()
        print(json.dumps({"summary": store.summary(),
                          "kind": net.kind,
                          "params": net.params(),
                          "metadata": net.metadata}, indent=2, default=str))
        return 0
    if args.action == "history":
        print(json.dumps(store.champion_meta(), indent=2, default=str))
        return 0
    if args.action == "rollback":
        meta = store.rollback(steps=args.steps)
        print(json.dumps({"rolled_back": True, "generation": meta.get("generation"),
                          "source": meta.get("source")}, indent=2))
        return 0
    if args.action == "promote":
        if not args.path:
            print("--path kandidat wajib untuk promote")
            return 2
        meta = store.promote(args.path, args.reason or "manual promote")
        print(json.dumps({"promoted": True, "generation": meta.get("generation"),
                          "source": meta.get("source")}, indent=2))
        return 0
    return 2


def cmd_lichess(args) -> int:
    from .lichess_bot import LichessBot
    cfg = _load_or_exit(args)
    if args.token:
        cfg = cfg.with_overrides("lichess", token=args.token)
    if args.max_games:
        cfg = cfg.with_overrides("lichess", max_games=args.max_games)
    if args.concurrency is not None:
        cfg = cfg.with_overrides("lichess", concurrency=args.concurrency)
    import os as _os
    token = cfg.lichess.token or _os.environ.get("KESTREL_LICHESS_TOKEN", "")
    if not token and not args.dry_run:
        print("Token Lichess belum ada. Set environment variable "
              "KESTREL_LICHESS_TOKEN atau --token. Buat token di "
              "https://lichess.org/account/oauth/token dengan scope bot:play. "
              "JANGAN menulis token ke config.toml.")
        return 2
    bot = LichessBot(cfg, token=token, dry_run=args.dry_run)
    try:
        bot.run()
        return 0
    except KeyboardInterrupt:
        print("\ndihentikan")
        return 0


def cmd_dashboard(args) -> int:
    import uvicorn
    uvicorn.run("kestrel.dashboard:app", host=args.host, port=args.port, reload=False)
    return 0


# ---------------- parser ----------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="kestrel",
        description="KESTREL 2.0 — evaluator AI TinyNNUE + training unified + cross-engine + bot Lichess")
    sub = ap.add_subparsers(dest="command")

    p = sub.add_parser("config", help="lihat/validasi/config template")
    p.add_argument("action", nargs="?", default="show",
                   choices=["show", "validate", "path", "init"])
    p.add_argument("--write", help="path output untuk init")
    p.add_argument("--config")
    p.set_defaults(func=cmd_config)

    nn = sub.add_parser("nnue", help="evaluator AI TinyNNUE: distill/info")
    p = nn
    p.add_argument("action", choices=["distill", "info"], nargs="?", default="info")
    p.add_argument("pgn", nargs="*", help="file PGN korpus (untuk distill)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--output", default=None, help="path checkpoint output")
    p.add_argument("--base-model", default=None, help="checkpoint warm start")
    p.add_argument("--config")
    p.set_defaults(func=cmd_nnue)

    p = sub.add_parser("train", help="pipeline unified: selfplay→knowledge→nnue→eval→promote")
    _add_common_overrides(p)
    p.add_argument("--generations", type=int, help="jumlah generasi training")
    p.add_argument("--eval-games", type=int, help="jumlah game verifikasi per generasi")
    p.add_argument("--eval-time-ms", type=int)
    p.add_argument("--quick", action="store_true", help="budget smoke-test")
    p.add_argument("--no-verify", action="store_true", help="tanpa match verifikasi")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("cross-train", help="training korpus Kestrel vs Stockfish (bukan self-play)")
    _add_common_overrides(p)
    p.add_argument("--generations", type=int)
    p.add_argument("--skill", type=int, default=5, help="Stockfish Skill Level awal")
    p.add_argument("--skill-min", type=int, default=0, help="batas bawah adaptive skill")
    p.add_argument("--skill-max", type=int, default=12, help="batas atas adaptive skill")
    p.add_argument("--parallel", type=int,
                   help="jumlah game cross yang dimainkan bersamaan (1 = sekuensial)")
    p.add_argument("--no-adaptive", action="store_true",
                   help="matikan adaptive skill (skill tetap konstan)")
    p.add_argument("--stockfish", help="path binary Stockfish (default: auto/PATH)")
    p.add_argument("--stockfish-time-ms", type=int,
                   help="waktu berpikir Stockfish per langkah (default: config [cross])")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.set_defaults(func=cmd_cross_train)

    p = sub.add_parser("selfplay", help="batch self-play saja (dengan book & champion)")
    _add_common_overrides(p)
    p.add_argument("--learn", action="store_true", help="perbarui knowledge base setelahnya")
    p.set_defaults(func=cmd_selfplay)

    p = sub.add_parser("eval", help="match detail champion vs kandidat/baseline")
    p.add_argument("--champion")
    p.add_argument("--challenger")
    p.add_argument("--eval-games", type=int)
    p.add_argument("--eval-time-ms", type=int)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--config")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("verify", help="bukti kekuatan: match vs baseline + benchmark penuh")
    p.add_argument("--challenger")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--config")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("bench", help="benchmark taktik/performa/konsistensi")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--workers", type=int, default=0,
                   help="N proses untuk metrik NPS agregat paralel (0 = lewati)")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("book", help="knowledge base: build/show/clear")
    p.add_argument("action", choices=["build", "show", "clear"])
    p.add_argument("--pgn", nargs="*", help="PGN tambahan untuk build")
    p.add_argument("--max-plies", type=int)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--config")
    p.set_defaults(func=cmd_book)

    p = sub.add_parser("model", help="checkpoint champion: show/history/rollback/promote")
    p.add_argument("action", choices=["show", "history", "rollback", "promote"])
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--path")
    p.add_argument("--reason")
    p.add_argument("--config")
    p.set_defaults(func=cmd_model)

    p = sub.add_parser("lichess", help="mainkan akun sebagai bot Lichess")
    p.add_argument("--token", help="token API lichess (atau set di config.toml)")
    p.add_argument("--max-games", type=int, help="batas jumlah game sesi ini")
    p.add_argument("--concurrency", type=int,
                   help="jumlah game Lichess yang dimainkan bersamaan")
    p.add_argument("--dry-run", action="store_true",
                   help="tanpa jaringan: tampilkan langkah yang akan dimainkan")
    p.add_argument("--config")
    p.set_defaults(func=cmd_lichess)

    p = sub.add_parser("dashboard", help="dashboard monitor localhost (opsional)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_dashboard)

    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "command", None):
        ap.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
