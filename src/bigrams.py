"""Bigram analysis over employee reviews.

A bigram is a pair of words that occur next to each other. Counting them tells you what
employees actually say about a subject rather than how often a subject is named: "pay"
alone is ambiguous, while "low pay" and "good pay" are not.

The published study reports bigrams separately for the pros and cons fields and for
current against former employees, and presents each as a word network. This module
produces both the counts and the network structure.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from preprocess import PreprocessConfig, tokenise_sentences


@dataclass(frozen=True)
class Bigram:
    a: str
    b: str
    count: int

    @property
    def phrase(self) -> str:
        return f"{self.a} {self.b}"


def count_bigrams(texts, config: PreprocessConfig | None = None,
                  min_count: int = 1) -> list[Bigram]:
    """Count adjacent word pairs across a set of reviews.

    Pairs are counted within sentences, so a pair never spans a sentence boundary. Pairs
    of the same word repeated are dropped, since they carry no relational meaning.
    """
    config = config or PreprocessConfig()
    counter: Counter = Counter()
    for text in texts:
        for sentence in tokenise_sentences(text, config):
            for first, second in zip(sentence, sentence[1:]):
                if first != second:
                    counter[(first, second)] += 1
    pairs = [Bigram(a, b, n) for (a, b), n in counter.items() if n >= min_count]
    # Sorted by count, then alphabetically, so ties resolve the same way on every run.
    pairs.sort(key=lambda p: (-p.count, p.a, p.b))
    return pairs


def top_bigrams(texts, top_n: int = 15, config: PreprocessConfig | None = None) -> list[Bigram]:
    """The n most frequent word pairs, which is what the study's figures report."""
    return count_bigrams(texts, config)[:top_n]


def to_network(pairs: list[Bigram]) -> dict:
    """Turn a bigram list into the node and edge structure the figures display.

    Nodes are words, edges are the pairs, and edge weight is the pair count. A word that
    appears in several pairs becomes a hub, which is how "work" and "good" come to sit at
    the centre of the published networks.
    """
    nodes: Counter = Counter()
    for p in pairs:
        nodes[p.a] += p.count
        nodes[p.b] += p.count
    return {
        "nodes": [{"word": w, "weight": n} for w, n in
                  sorted(nodes.items(), key=lambda kv: (-kv[1], kv[0]))],
        "edges": [{"a": p.a, "b": p.b, "count": p.count} for p in pairs],
    }


def band_edges(pairs: list[Bigram], bands: int = 5) -> list[dict]:
    """Assign each pair to a weight band.

    The published figures encode edge frequency as a banded colour scale rather than a
    printed number. Banding a recomputed set the same way makes the two directly
    comparable.
    """
    if not pairs:
        return []
    high = max(p.count for p in pairs)
    low = min(p.count for p in pairs)
    span = max(high - low, 1)
    out = []
    for p in pairs:
        band = 1 + int((p.count - low) / span * (bands - 1) + 0.5)
        out.append({"a": p.a, "b": p.b, "count": p.count, "weight": band})
    return out


def compare_groups(groups: dict[str, list[str]], top_n: int = 15,
                   config: PreprocessConfig | None = None) -> dict[str, dict]:
    """Run the bigram analysis separately for each group of reviews.

    The study splits four ways: pros and cons, each by current and former employees.
    Passing those four sets here reproduces that split.
    """
    return {
        name: {
            "reviews": len(texts),
            "top_bigrams": [{"pair": b.phrase, "count": b.count}
                            for b in top_bigrams(texts, top_n, config)],
            "network": to_network(top_bigrams(texts, top_n, config)),
        }
        for name, texts in groups.items()
    }
