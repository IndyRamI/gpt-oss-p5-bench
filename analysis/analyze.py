"""Aggregate a run directory into summary.csv, plots, and REPORT.md.

    python analysis/analyze.py results/<RUN_ID>
"""

import csv
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

# Fixed topology -> color mapping (validated categorical palette, fixed order):
# a topology keeps its color across every chart and every run.
TOPO_ORDER = ["tp1x4", "tp2x2", "tp4x1", "tp1x8", "tp2x4", "tp4x2", "tp8x1"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
COLOR = dict(zip(TOPO_ORDER, PALETTE))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titlecolor": INK, "lines.linewidth": 2,
    "lines.markersize": 6, "legend.frameon": False,
})


def g(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def gpu_stats(path):
    if not os.path.exists(path):
        return {}
    util, power = [], []
    with open(path) as f:
        for row in csv.DictReader((l.replace(", ", ",") for l in f)):
            try:
                u = float(row["utilization.gpu [%]"].rstrip(" %"))
                p = float(row["power.draw [W]"].rstrip(" W"))
            except (KeyError, ValueError):
                continue
            if u > 0:  # only GPUs actually in use
                util.append(u)
                power.append(p)
    if not util:
        return {}
    return {"gpu_util_avg": sum(util) / len(util), "gpu_power_avg_w": sum(power) / len(power)}


def load_rows(run_dir):
    rows, servers = [], {}
    for phase_dir in sorted(glob.glob(os.path.join(run_dir, "*", "*"))):
        topo, phase = phase_dir.split(os.sep)[-2:]
        srv_path = os.path.join(phase_dir, "server.json")
        srv = json.load(open(srv_path)) if os.path.exists(srv_path) else {}
        servers[(topo, phase)] = srv
        gpus = srv.get("budget") or (srv.get("tp", 1) * srv.get("replicas", 1))
        gstats = gpu_stats(os.path.join(phase_dir, "gpu.csv"))
        for f in glob.glob(os.path.join(phase_dir, f"{phase}_*.json")):
            res = json.load(open(f))
            s, m = res["summary"], res["meta"]
            row = {
                "topology": topo, "phase": phase, "tp": srv.get("tp"),
                "replicas": srv.get("replicas"), "gpus": gpus,
                "level_kind": m["level_kind"], "level": m["level"],
                "ok": s.get("ok"), "errors": s.get("errors"),
                "ttft_p50_ms": g(s, "ttft_ms", "p50"), "ttft_p90_ms": g(s, "ttft_ms", "p90"),
                "ttft_p99_ms": g(s, "ttft_ms", "p99"),
                "prompt_tokens_mean": g(s, "prompt_tokens", "mean"),
                **gstats,
            }
            if phase == "prefill":
                row["input_tok_s"] = s.get("input_tok_s")
                row["input_tok_s_per_gpu"] = (s.get("input_tok_s") or 0) / gpus if gpus else None
                row["req_s"] = s.get("req_s")
            else:
                row.update({
                    "decode_tok_s": s.get("decode_tok_s"),
                    "decode_tok_s_per_gpu": (s.get("decode_tok_s") or 0) / gpus if gpus else None,
                    "tpot_p50_ms": g(s, "tpot_ms", "p50"), "tpot_p99_ms": g(s, "tpot_ms", "p99"),
                    "itl_p99_ms": g(s, "itl_ms", "p99"),
                    "per_user_tok_s": s.get("per_user_tok_s"),
                    "effective_batch": s.get("effective_batch"),
                    "window_clean": s.get("window_clean"),
                    "unclean_reason": s.get("unclean_reason"),
                })
            rows.append(row)
    key = lambda r: (TOPO_ORDER.index(r["topology"]) if r["topology"] in TOPO_ORDER else 99,
                     r["phase"], r["level"])
    return sorted(rows, key=key), servers


def series(rows, phase, ycol):
    out = {}
    for r in rows:
        if r["phase"] == phase and r.get(ycol) is not None and r["level_kind"] == "concurrency":
            out.setdefault(r["topology"], []).append(r)
    return out


def line_panel(ax, data, xcol, ycol, title, ylabel, logx=True, hollow_unclean=False):
    for topo, rs in data.items():
        xs, ys = [r[xcol] for r in rs], [r[ycol] for r in rs]
        c = COLOR.get(topo, INK2)
        ax.plot(xs, ys, color=c, label=topo, marker="o", markersize=7,
                markeredgecolor="#fcfcfb", markeredgewidth=2)
        if hollow_unclean:
            bad = [(x, y) for x, y, r in zip(xs, ys, rs) if r.get("window_clean") is False]
            if bad:
                ax.scatter(*zip(*bad), s=60, facecolor="#fcfcfb", edgecolor=c,
                           linewidth=2, zorder=3)
        ax.annotate(topo, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK2, va="center", fontsize=9)
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({r[xcol] for rs in data.values() for r in rs}))
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_title(title, loc="left")
    ax.set_xlabel(xcol.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)


def plot_all(rows, run_dir):
    figs = []
    pre = series(rows, "prefill", "ttft_p50_ms")
    if pre:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        line_panel(axes[0], pre, "level", "ttft_p50_ms", "TTFT p50 (~15K-token prompt)", "ms")
        line_panel(axes[1], pre, "level", "ttft_p99_ms", "TTFT p99", "ms")
        line_panel(axes[2], series(rows, "prefill", "input_tok_s_per_gpu"), "level",
                   "input_tok_s_per_gpu", "Prefill throughput per GPU", "input tokens/s/GPU")
        for ax in axes:
            ax.set_xlabel("concurrent requests (total)")
        axes[0].legend(loc="upper left")
        fig.tight_layout()
        figs.append(("prefill.png", fig))

    dec = series(rows, "decode", "decode_tok_s_per_gpu")
    if dec:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        line_panel(axes[0], dec, "level", "decode_tok_s_per_gpu",
                   "Decode throughput per GPU (hollow = no clean decode window)",
                   "output tokens/s/GPU", hollow_unclean=True)
        line_panel(axes[1], series(rows, "decode", "tpot_p50_ms"), "level", "tpot_p50_ms",
                   "Time per output token p50", "ms", hollow_unclean=True)
        for ax in axes:
            ax.set_xlabel("concurrent sequences (total)")
        axes[0].legend(loc="upper left")
        fig.tight_layout()
        figs.append(("decode.png", fig))

        # Pareto: interactivity (x) vs efficiency (y)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for topo, rs in series(rows, "decode", "per_user_tok_s").items():
            rs = sorted(rs, key=lambda r: r["per_user_tok_s"])
            ax.plot([r["per_user_tok_s"] for r in rs], [r["decode_tok_s_per_gpu"] for r in rs],
                    color=COLOR.get(topo, INK2), label=topo, marker="o", markersize=7,
                    markeredgecolor="#fcfcfb", markeredgewidth=2)
            for r in rs:
                ax.annotate(f"c{r['level']}", (r["per_user_tok_s"], r["decode_tok_s_per_gpu"]),
                            xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK2)
        ax.set_title("Decode Pareto @15K context: up-and-right is better", loc="left")
        ax.set_xlabel("per-user tokens/s (1 / TPOT p50)")
        ax.set_ylabel("output tokens/s per GPU")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        fig.tight_layout()
        figs.append(("decode_pareto.png", fig))

    for name, fig in figs:
        fig.savefig(os.path.join(run_dir, name), dpi=140)
        plt.close(fig)
    return [n for n, _ in figs]


def fmt(v, nd=0):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "**no**"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def table(rows, cols):
    head = "| " + " | ".join(c for c, _, _ in cols) + " |"
    sep = "|" + "|".join("---:" for _ in cols) + "|"
    body = ["| " + " | ".join(fmt(r.get(k), nd) for _, k, nd in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def findings(rows):
    out = []
    pre = [r for r in rows if r["phase"] == "prefill" and r["ttft_p50_ms"]]
    dec = [r for r in rows if r["phase"] == "decode" and r.get("decode_tok_s_per_gpu")]
    if pre:
        lo = min(r["level"] for r in pre)
        b = min((r for r in pre if r["level"] == lo), key=lambda r: r["ttft_p50_ms"])
        out.append(f"Lowest single-request TTFT: **{b['topology']}** at {b['ttft_p50_ms']:,.0f} ms (c={lo}).")
        b = max(pre, key=lambda r: r["input_tok_s_per_gpu"] or 0)
        out.append(f"Best prefill efficiency: **{b['topology']}** at {b['input_tok_s_per_gpu']:,.0f} input tok/s/GPU (c={b['level']}).")
    if dec:
        b = max(dec, key=lambda r: r["decode_tok_s_per_gpu"])
        out.append(f"Best decode efficiency: **{b['topology']}** at {b['decode_tok_s_per_gpu']:,.0f} tok/s/GPU (c={b['level']}, TPOT p50 {fmt(b['tpot_p50_ms'], 1)} ms).")
        b = max(dec, key=lambda r: r.get("per_user_tok_s") or 0)
        out.append(f"Best per-user decode speed: **{b['topology']}** at {fmt(b['per_user_tok_s'])} tok/s (c={b['level']}).")
        na = [r for r in dec if r.get("unclean_reason") == "not_admitted"]
        short = [r for r in dec if r.get("unclean_reason") == "window_too_short"]
        if na:
            out.append("Server could not run the whole batch at once (KV capacity, max_num_seqs or "
                       "prefill backlog), so prefill and decode overlapped: "
                       + ", ".join(f"{r['topology']}@c{r['level']} (ran {r['effective_batch']})" for r in na)
                       + ". Compare with the capacity table.")
        if short:
            out.append("Decode window too short to measure cleanly (raise output_len): "
                       + ", ".join(f"{r['topology']}@c{r['level']}" for r in short) + ".")
    return out


def main():
    run_dir = sys.argv[1].rstrip("/")
    rows, servers = load_rows(run_dir)
    if not rows:
        sys.exit(f"no results under {run_dir}")

    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(os.path.join(run_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    figs = plot_all(rows, run_dir)
    srv_rows = [{"topology": t, "phase": p, **{k: v for k, v in s.items() if k != "engine_kwargs"},
                 "kv_tokens_per_replica": (min(s["kv_cache_tokens"]) if s.get("kv_cache_tokens") else None)}
                for (t, p), s in servers.items() if s]
    for r in srv_rows:
        kv = r["kv_tokens_per_replica"]
        r["fits_15k_seqs_per_replica"] = (kv // 15500) if kv else None
        r["fits_15k_seqs_total"] = (kv // 15500) * r["replicas"] if kv else None

    any_srv = next((s for s in servers.values() if s), {})
    md = [f"# gpt-oss-120b on P5 - run `{os.path.basename(run_dir)}`", "",
          f"Ray {any_srv.get('ray_version', '?')} / vLLM {any_srv.get('vllm_version', '?')}. "
          "Raw data: `summary.csv` and per-level JSON under each `<topology>/<phase>/`.", "",
          "## Key findings (auto-generated - read alongside the charts)", ""]
    md += [f"- {x}" for x in findings(rows)] + ["", "## Deployed capacity", "",
           table(srv_rows, [("topology", "topology", 0), ("phase", "phase", 0), ("TP", "tp", 0),
                            ("replicas", "replicas", 0), ("KV tokens/replica", "kv_tokens_per_replica", 0),
                            ("~15K seqs/replica", "fits_15k_seqs_per_replica", 0),
                            ("15K seqs total", "fits_15k_seqs_total", 0),
                            ("startup s", "startup_s", 0)]), ""]
    if "prefill.png" in figs:
        md += ["## Prefill (15K in, 1 out)", "", "![prefill](prefill.png)", "",
               table([r for r in rows if r["phase"] == "prefill"],
                     [("topology", "topology", 0), ("conc", "level", 0), ("TTFT p50 ms", "ttft_p50_ms", 0),
                      ("TTFT p90 ms", "ttft_p90_ms", 0), ("TTFT p99 ms", "ttft_p99_ms", 0),
                      ("input tok/s", "input_tok_s", 0), ("tok/s/GPU", "input_tok_s_per_gpu", 0),
                      ("GPU util %", "gpu_util_avg", 0), ("errors", "errors", 0)]), ""]
    if "decode.png" in figs:
        md += ["## Decode (15K context, forced long output)", "",
               "![decode](decode.png)", "", "![pareto](decode_pareto.png)", "",
               table([r for r in rows if r["phase"] == "decode"],
                     [("topology", "topology", 0), ("conc", "level", 0), ("eff. batch", "effective_batch", 0),
                      ("clean window", "window_clean", 0), ("decode tok/s", "decode_tok_s", 0),
                      ("tok/s/GPU", "decode_tok_s_per_gpu", 0), ("TPOT p50 ms", "tpot_p50_ms", 1),
                      ("TPOT p99 ms", "tpot_p99_ms", 1), ("ITL p99 ms", "itl_p99_ms", 1),
                      ("per-user tok/s", "per_user_tok_s", 0), ("TTFT p50 ms", "ttft_p50_ms", 0),
                      ("errors", "errors", 0)]), ""]
    md += ["## Analysis notes", "", "_Fill in: interpretation, anomalies, what to try next._", ""]
    with open(os.path.join(run_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(md))
    print(f"wrote {run_dir}/REPORT.md, summary.csv, {', '.join(figs)}")


if __name__ == "__main__":
    main()
