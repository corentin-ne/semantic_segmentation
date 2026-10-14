"""Training & results dashboard.

    streamlit run dashboard/app.py -- --runs runs

Reads the run directories written by ``semseg train`` (JSON/PNG only), refreshes live while
a training is in progress, and can run a trained checkpoint on your own images.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STATUS_ICON = {"running": "🟢", "finished": "✅", "early_stopped": "⏹️", "interrupted": "⏸️",
               "failed": "❌", "starting": "⏳"}
METRIC_LABELS = {"mIoU": "mIoU", "val_loss": "Loss validation", "train_loss": "Loss entraînement",
                 "pixel_acc": "Pixel accuracy", "mean_acc": "Mean class accuracy", "fwIoU": "Frequency-weighted IoU",
                 "mDice": "mDice", "lr": "Learning rate", "train_img_per_s": "Débit (img/s)"}
PLOT = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), height=380)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=os.environ.get("SEMSEG_RUNS", "runs"))
    args, _ = parser.parse_known_args()
    return args


# ---------------------------------------------------------------- data loading
def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # line being written right now
            pass
    return pd.DataFrame(rows)


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _fingerprint(runs_dir: Path) -> tuple:
    return tuple(sorted((str(p), p.stat().st_mtime) for p in runs_dir.glob("*/summary.json")))


@st.cache_data(show_spinner=False)
def load_runs(runs_dir: str, _fingerprint: tuple) -> list[dict]:
    runs = []
    for d in sorted(Path(runs_dir).glob("*/summary.json")):
        d = d.parent
        summary = _read_json(d / "summary.json", {})
        runs.append({
            "name": d.name, "dir": str(d), "summary": summary,
            "config": _read_json(d / "config.json", {}),
            "metrics": _read_jsonl(d / "metrics.jsonl"),
            "steps": _read_jsonl(d / "steps.jsonl"),
            "confusion": _read_json(d / "confusion.json"),
            "samples": sorted(str(p) for p in (d / "samples").glob("epoch_*.png")),
            "checkpoint": str(d / "checkpoints" / "best.pt") if (d / "checkpoints" / "best.pt").exists() else None,
        })
    return runs


@st.cache_data(show_spinner=False)
def load_benchmarks(runs_dir: str) -> pd.DataFrame:
    frames = []
    for path in Path(runs_dir).glob("benchmark*.json"):
        data = _read_json(path, {})
        df = pd.DataFrame(data.get("results", []))
        if not df.empty:
            df["source"] = f"{path.stem} · {data.get('device')} · {data.get('size')}px"
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_resource(show_spinner="Chargement du modèle…")
def get_predictor(checkpoint: str):
    from semseg.inference import Predictor
    return Predictor(checkpoint, device="auto")


# ---------------------------------------------------------------- views
def runs_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        s, m = r["summary"], r["metrics"]
        last = m.iloc[-1] if not m.empty else {}
        rows.append({
            "": STATUS_ICON.get(s.get("status"), "•"),
            "Run": r["name"], "Modèle": s.get("model"), "Dataset": s.get("dataset"),
            "Époque": f"{s.get('epoch', 0)}/{s.get('epochs', '?')}",
            "Meilleur mIoU": s.get("best_mIoU"), "Pixel acc": last.get("pixel_acc") if len(m) else None,
            "Params (M)": s.get("params_m"), "GFLOPs": s.get("gflops"),
            "img/s (train)": last.get("train_img_per_s") if len(m) else None,
            "Device": s.get("device"),
        })
    return pd.DataFrame(rows)


def view_overview(runs: list[dict]) -> None:
    finished = [r for r in runs if r["summary"].get("best_mIoU") is not None]
    if finished:
        best = max(finished, key=lambda r: r["summary"]["best_mIoU"])
        lightest = min(finished, key=lambda r: r["summary"].get("gflops") or 1e9)
        c = st.columns(4)
        c[0].metric("Runs", len(runs), f"{sum(r['summary'].get('status') == 'running' for r in runs)} en cours")
        c[1].metric("Meilleur mIoU", f"{best['summary']['best_mIoU'] * 100:.1f} %", best["summary"].get("model"),
                    delta_color="off")
        c[2].metric("Le plus léger", f"{lightest['summary'].get('gflops')} GFLOPs", lightest["summary"].get("model"),
                    delta_color="off")
        c[3].metric("Images (train / val)",
                    f"{best['summary'].get('train_images', '?')} / {best['summary'].get('val_images', '?')}")
    st.dataframe(
        runs_table(runs), hide_index=True, use_container_width=True,
        column_config={
            "Meilleur mIoU": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
            "Pixel acc": st.column_config.NumberColumn(format="%.3f"),
        },
    )


def view_training(runs: list[dict]) -> None:
    metrics = [k for k in METRIC_LABELS if any(k in r["metrics"] for r in runs)]
    if not metrics:
        st.info("Aucune époque évaluée pour l'instant.")
        return
    c1, c2 = st.columns([1, 1])
    left = c1.selectbox("Métrique", metrics, format_func=METRIC_LABELS.get, index=metrics.index("mIoU") if "mIoU" in metrics else 0)
    right = c2.selectbox("Comparer avec", metrics, format_func=METRIC_LABELS.get,
                         index=metrics.index("val_loss") if "val_loss" in metrics else 0)
    c1, c2 = st.columns(2)
    for col, metric in ((c1, left), (c2, right)):
        fig = go.Figure()
        for r in runs:
            if metric in r["metrics"]:
                fig.add_scatter(x=r["metrics"]["epoch"], y=r["metrics"][metric], mode="lines+markers", name=r["name"])
        fig.update_layout(title=METRIC_LABELS[metric], xaxis_title="époque", showlegend=True,
                          legend=dict(orientation="h", y=-0.25), **PLOT)
        col.plotly_chart(fig, use_container_width=True)

    steps = [r for r in runs if not r["steps"].empty]
    if steps:
        smooth = st.slider("Lissage de la loss par itération", 0.0, 0.99, 0.8, 0.01)
        fig = go.Figure()
        for r in steps:
            s = r["steps"]
            fig.add_scatter(x=s["epoch"], y=s["loss"].ewm(alpha=1 - smooth).mean() if smooth else s["loss"],
                            mode="lines", name=r["name"])
        fig.update_layout(title="Loss d'entraînement (par itération)", xaxis_title="époque",
                          yaxis_type="log", legend=dict(orientation="h", y=-0.25), **PLOT)
        st.plotly_chart(fig, use_container_width=True)


def view_results(runs: list[dict]) -> None:
    candidates = [r for r in runs if not r["metrics"].empty]
    if not candidates:
        st.info("Pas encore de résultats.")
        return
    run = st.selectbox("Run", candidates, format_func=lambda r: r["name"])
    names = run["summary"].get("class_names") or []
    best = run["summary"].get("best", {})
    if best:
        c = st.columns(5)
        for col, key in zip(c, ("mIoU", "pixel_acc", "mean_acc", "fwIoU", "mDice")):
            col.metric(METRIC_LABELS.get(key, key), f"{best.get(key, 0) * 100:.1f} %")
        st.caption(f"Meilleure époque : {run['summary'].get('best_epoch')}")

    c1, c2 = st.columns(2)
    per_class = best.get("per_class_iou")
    if per_class and names:
        df = pd.DataFrame({"classe": names, "IoU": per_class}).dropna().sort_values("IoU")
        fig = px.bar(df, x="IoU", y="classe", orientation="h", range_x=[0, 1], color="IoU",
                     color_continuous_scale="Tealgrn")
        fig.update_layout(title="IoU par classe (meilleur checkpoint)", coloraxis_showscale=False,
                          **{**PLOT, "height": max(320, 22 * len(df))})
        c1.plotly_chart(fig, use_container_width=True)
    if run["confusion"] and names:
        cm = np.asarray(run["confusion"], dtype=float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig = px.imshow(norm, x=names, y=names, color_continuous_scale="Blues", zmin=0, zmax=1,
                        labels=dict(x="prédiction", y="vérité terrain", color="taux"))
        fig.update_layout(title="Matrice de confusion (normalisée par ligne)",
                          **{**PLOT, "height": max(360, 22 * len(names) + 120)})
        c2.plotly_chart(fig, use_container_width=True)

    if run["samples"]:
        idx = st.select_slider("Prédictions sur la validation — époque", options=list(range(len(run["samples"]))),
                               value=len(run["samples"]) - 1,
                               format_func=lambda i: Path(run["samples"][i]).stem.split("_")[1].lstrip("0"))
        st.image(run["samples"][idx], use_container_width=False)


def view_compare(runs: list[dict], runs_dir: str) -> None:
    table = runs_table(runs).dropna(subset=["Meilleur mIoU"])
    if not table.empty:
        fig = px.scatter(table, x="GFLOPs", y="Meilleur mIoU", size="Params (M)", color="Modèle",
                         hover_name="Run", size_max=40)
        fig.update_layout(title="Précision vs coût de calcul (taille = paramètres)", **PLOT)
        st.plotly_chart(fig, use_container_width=True)
    bench = load_benchmarks(runs_dir)
    if bench.empty:
        st.info("Lance `python -m semseg benchmark --output runs/benchmark.json` pour mesurer la latence de chaque modèle.")
        return
    source = st.selectbox("Benchmark", sorted(bench["source"].unique()))
    b = bench[bench["source"] == source].sort_values("latency_ms")
    c1, c2 = st.columns(2)
    fig = px.bar(b, x="latency_ms", y="model", orientation="h", color="family", text="latency_ms")
    fig.update_traces(texttemplate="%{text:.0f} ms")
    fig.update_layout(title="Latence (médiane, batch 1)", **PLOT)
    c1.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(b, x="gflops", y="params_m", text="model", color="family", log_x=True)
    fig.update_traces(textposition="top center")
    fig.update_layout(title="GFLOPs vs paramètres (M)", **PLOT)
    c2.plotly_chart(fig, use_container_width=True)
    st.dataframe(b.drop(columns=["source"]), hide_index=True, use_container_width=True)


def view_playground(runs: list[dict]) -> None:
    from semseg.visualize import colorize, overlay

    ready = [r for r in runs if r["checkpoint"]]
    if not ready:
        st.info("Aucun checkpoint disponible : entraîne un modèle d'abord.")
        return
    run = st.selectbox("Modèle entraîné", ready, format_func=lambda r: f"{r['name']}  (mIoU {r['summary'].get('best_mIoU', 0):.3f})")
    predictor = get_predictor(run["checkpoint"])
    c1, c2, c3 = st.columns([2, 1, 1])
    upload = c1.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"])
    tta = c2.toggle("TTA (flip + multi-échelle)")
    alpha = c3.slider("Opacité du masque", 0.0, 1.0, 0.55)
    if upload is not None:
        image = Image.open(upload).convert("RGB")
    elif predictor.meta["dataset"] == "synthetic":
        from semseg.data.synthetic import render_sample
        seed = c1.number_input("…ou image synthétique n°", 0, 10_000, 3)
        image = render_sample(10_000_000 + int(seed), 256)[0]
    else:
        st.info("Dépose une image pour la segmenter.")
        return
    labels, conf = predictor.predict(image, tta=tta)
    rgb = np.asarray(image)
    color = colorize(labels, predictor.palette)
    c1, c2, c3 = st.columns(3)
    c1.image(rgb, caption="Image", use_container_width=True)
    c2.image(overlay(rgb, color, alpha), caption="Segmentation", use_container_width=True)
    c3.image((conf * 255).astype(np.uint8), caption="Confiance (blanc = sûr)", use_container_width=True)
    ids, counts = np.unique(labels, return_counts=True)
    df = pd.DataFrame({"classe": [predictor.class_names[i] for i in ids], "% pixels": counts / counts.sum() * 100,
                       "couleur": [f"rgb{tuple(int(v) for v in predictor.palette[i])}" for i in ids]})
    fig = px.bar(df.sort_values("% pixels"), x="% pixels", y="classe", orientation="h")
    fig.update_traces(marker_color=df.sort_values("% pixels")["couleur"].tolist())
    fig.update_layout(title="Classes détectées", **{**PLOT, "height": 80 + 30 * len(df)})
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- page
def _selected(runs_dir: str, chosen: list[str]) -> list[dict]:
    return [r for r in load_runs(runs_dir, _fingerprint(Path(runs_dir))) if r["name"] in chosen]


def main() -> None:
    st.set_page_config(page_title="Semantic Segmentation", page_icon="🧩", layout="wide")
    args = parse_args()
    with st.sidebar:
        st.title("🧩 Semantic Segmentation")
        st.caption("Pascal VOC · ENSC — Corentin & Thomas")
        runs_dir = st.text_input("Dossier des runs", args.runs)
        live = st.toggle("Rafraîchissement auto (10 s)", value=True)
    if not Path(runs_dir).exists():
        st.warning(f"Dossier `{runs_dir}` introuvable. Lance un entraînement : `python -m semseg train -c configs/synthetic_demo.yaml`")
        return

    all_runs = load_runs(runs_dir, _fingerprint(Path(runs_dir)))
    with st.sidebar:
        chosen = st.multiselect("Runs affichés", [r["name"] for r in all_runs], default=[r["name"] for r in all_runs])
    runs = [r for r in all_runs if r["name"] in chosen]

    tabs = st.tabs(["📋 Vue d'ensemble", "📈 Entraînement", "🎯 Résultats", "⚖️ Comparaison", "🖼️ Démo"])

    every = "10s" if live else None

    @st.fragment(run_every=every)
    def live_overview():
        view_overview(_selected(runs_dir, chosen))

    @st.fragment(run_every=every)
    def live_training():
        view_training(_selected(runs_dir, chosen))

    with tabs[0]:
        live_overview()
    with tabs[1]:
        live_training()
    with tabs[2]:
        view_results(runs)
    with tabs[3]:
        view_compare(runs, runs_dir)
    with tabs[4]:
        view_playground(runs)


main()
