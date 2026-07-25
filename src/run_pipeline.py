"""Run the full analysis chain on a review corpus.

Steps, in order:
    1. Drop duplicates and empty reviews.
    2. Count bigrams for pros and cons, split by employment status, and build the word
       network for each of the four groups.
    3. Fit topic models across a range of topic counts, scoring each on semantic
       coherence and exclusivity.
    4. Fit the selected model and report its topics.
    5. Profile topic prevalence against employment status, year and rating.

Usage:
    python src/run_pipeline.py --input data/synthetic_reviews.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bigrams import band_edges, top_bigrams, to_network
from preprocess import PreprocessConfig, deduplicate, is_usable, tokenise
from topics import (build_matrix, document_topics, fit, proportions_by_covariate,
                    search_k, suggest_k, top_words, topic_proportions)

ROOT = Path(__file__).resolve().parents[1]


def run(input_path: Path, k_min: int, k_max: int, k_override: int | None,
        top_n: int, seed: int) -> dict:
    df = pd.read_csv(input_path)
    config = PreprocessConfig()

    for column in ("review", "pros", "cons"):
        if column not in df.columns:
            df[column] = ""
    df["combined_text"] = (
        df.get("review_title", "").fillna("").astype(str) + " . "
        + df["review"].fillna("").astype(str) + " . "
        + df["pros"].fillna("").astype(str) + " . "
        + df["cons"].fillna("").astype(str)
    )

    rows_in = len(df)
    df = df.iloc[deduplicate(df["combined_text"].tolist())].reset_index(drop=True)
    df = df[df["combined_text"].apply(lambda t: is_usable(t, config))].reset_index(drop=True)

    # ---- bigram networks, four ways ----
    networks = {}
    statuses = sorted(df["job_status"].dropna().unique()) if "job_status" in df else []
    for field, label in (("pros", "pros"), ("cons", "cons")):
        for status in statuses:
            subset = df[df["job_status"] == status][field].fillna("").astype(str)
            pairs = top_bigrams(subset.tolist(), top_n, config)
            key = f"{label}_{'current' if 'Current' in status else 'former'}"
            networks[key] = {
                "sentiment": label,
                "status": status,
                "reviews": int(len(subset)),
                "top_bigrams": [{"pair": b.phrase, "count": b.count} for b in pairs],
                "edges": band_edges(pairs),
                "nodes": to_network(pairs)["nodes"],
            }

    # ---- topic model ----
    documents = [tokenise(t, config) for t in df["combined_text"]]
    matrix, vocabulary = build_matrix(documents)
    fit_rows = search_k(matrix, vocabulary, range(k_min, k_max + 1), seed)
    k = k_override or suggest_k(fit_rows)

    model = fit(matrix, k, seed)
    doc_topics = document_topics(model, matrix)
    words = top_words(model, vocabulary)
    proportions = topic_proportions(doc_topics)

    topics = [
        {
            "topic": i + 1,
            "proportion_pct": round(100 * proportions[i], 1),
            "top_words": words[i],
        }
        for i in range(k)
    ]
    topics.sort(key=lambda t: -t["proportion_pct"])

    covariates = {}
    for name, column in (("status", "job_status"), ("year", "year"), ("rating", "rating")):
        if column in df.columns:
            covariates[name] = proportions_by_covariate(doc_topics, df[column].tolist())

    return {
        "input": input_path.name,
        "corpus": {
            "rows_in": int(rows_in),
            "rows_analysed": int(len(df)),
            "retention_pct": round(100 * len(df) / rows_in, 1),
            "vocabulary": int(len(vocabulary)),
        },
        "bigram_networks": networks,
        "topic_fit": [r.as_dict() for r in fit_rows],
        "selected_k": int(k),
        "topics": topics,
        "topic_proportions_by_covariate": covariates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "synthetic_reviews.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "pipeline_run.json")
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--k", type=int, default=None, help="fix the topic count instead of suggesting one")
    parser.add_argument("--top-n", type=int, default=15, help="bigrams per network")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = run(args.input, args.k_min, args.k_max, args.k, args.top_n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"analysed {payload['corpus']['rows_analysed']} reviews "
          f"({payload['corpus']['vocabulary']} terms)")
    print(f"selected {payload['selected_k']} topics")
    for t in payload["topics"]:
        print(f"  {t['proportion_pct']:>5.1f}%  {', '.join(t['top_words'][:7])}")
    for key, net in payload["bigram_networks"].items():
        lead = net["top_bigrams"][0] if net["top_bigrams"] else {"pair": "-", "count": 0}
        print(f"  {key:<14} top pair: {lead['pair']} ({lead['count']})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
