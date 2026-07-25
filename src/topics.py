"""Topic modelling with the diagnostics the published study used to choose K.

A note on what this module is and is not. The published analysis used Structural Topic
Modeling through the R package `stm`, whose distinguishing feature is that document
metadata enters the prior on topic prevalence. There is no equivalent implementation in
Python. This module therefore fits Latent Dirichlet Allocation, which STM extends, and
profiles topic prevalence against the covariates afterwards rather than inside the model.

Everything else follows the study: candidate models are compared on semantic coherence
and exclusivity, the search runs over four to ten topics, and the final label for each
topic is a human judgement rather than a model output.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class TopicFit:
    k: int
    semantic_coherence: float
    exclusivity: float
    perplexity: float

    def as_dict(self) -> dict:
        return asdict(self)


def build_matrix(documents, min_df: int = 5, max_df: float = 0.5,
                 max_features: int | None = 5000):
    """Document-term matrix over pre-tokenised documents."""
    vectoriser = CountVectorizer(
        analyzer=lambda d: d, min_df=min_df, max_df=max_df, max_features=max_features
    )
    matrix = vectoriser.fit_transform(documents)
    return matrix, vectoriser.get_feature_names_out()


def top_words(model: LatentDirichletAllocation, vocabulary, n: int = 12) -> list[list[str]]:
    """The highest-probability words for each topic."""
    return [[vocabulary[i] for i in topic.argsort()[::-1][:n]]
            for topic in model.components_]


def semantic_coherence(model: LatentDirichletAllocation, matrix, vocabulary,
                       n: int = 10) -> float:
    """Mimno's semantic coherence, averaged across topics.

    For each topic, every pair of its top words is scored by how often the rarer word
    appears in documents that also contain the commoner one. Topics whose leading words
    genuinely travel together score higher. The measure falls as K rises, which is why it
    is never used on its own.
    """
    binary = (matrix > 0).astype(int)
    index = {w: i for i, w in enumerate(vocabulary)}
    scores = []
    for words in top_words(model, vocabulary, n):
        cols = [index[w] for w in words if w in index]
        total = 0.0
        for i in range(1, len(cols)):
            for j in range(i):
                both = int(binary[:, cols[i]].multiply(binary[:, cols[j]]).sum())
                alone = int(binary[:, cols[j]].sum())
                total += np.log((both + 1) / max(alone, 1))
        scores.append(total)
    return float(np.mean(scores)) if scores else 0.0


def exclusivity(model: LatentDirichletAllocation, vocabulary, n: int = 10) -> float:
    """Exclusivity of the top words, averaged across topics.

    A word is exclusive when its probability is concentrated in one topic rather than
    spread across several. Exclusivity rises as K rises, so it pulls against semantic
    coherence, and the pair together bracket a sensible range for K.
    """
    weights = model.components_ / model.components_.sum(axis=1, keepdims=True)
    column_totals = weights.sum(axis=0)
    scores = []
    for t in range(weights.shape[0]):
        top = weights[t].argsort()[::-1][:n]
        scores.append(float(np.mean(weights[t][top] / column_totals[top])))
    return float(np.mean(scores))


def fit(matrix, k: int, seed: int = 42, max_iter: int = 40) -> LatentDirichletAllocation:
    # A sparse document prior suits short reviews, most of which sit on one or two
    # themes rather than spreading evenly across all of them.
    model = LatentDirichletAllocation(
        n_components=k,
        learning_method="batch",
        max_iter=max_iter,
        random_state=seed,
        doc_topic_prior=1 / k,
        topic_word_prior=0.01,
    )
    model.fit(matrix)
    return model


def search_k(matrix, vocabulary, k_range=range(4, 11), seed: int = 42) -> list[TopicFit]:
    """Fit every candidate model and score it on both diagnostics.

    The study searched four to ten topics, which is the default here.
    """
    rows: list[TopicFit] = []
    for k in k_range:
        model = fit(matrix, k, seed)
        rows.append(
            TopicFit(
                k=k,
                semantic_coherence=round(semantic_coherence(model, matrix, vocabulary), 2),
                exclusivity=round(exclusivity(model, vocabulary), 4),
                perplexity=round(float(model.perplexity(matrix)), 1),
            )
        )
    return rows


def suggest_k(rows: list[TopicFit]) -> int:
    """Suggest a topic count from the two diagnostics.

    Both measures are rescaled to a common range and summed, which picks the solution
    with the best balance between them. The study treated this as one input among
    several: the author also read the topics and judged whether they were interpretable,
    and that judgement is not something this function can make.
    """
    if not rows:
        raise ValueError("no candidate models to compare")
    coherence = np.array([r.semantic_coherence for r in rows], dtype=float)
    exclusive = np.array([r.exclusivity for r in rows], dtype=float)

    def rescale(values):
        span = values.max() - values.min()
        return np.zeros_like(values) if span == 0 else (values - values.min()) / span

    combined = rescale(coherence) + rescale(exclusive)
    return rows[int(combined.argmax())].k


def document_topics(model: LatentDirichletAllocation, matrix) -> np.ndarray:
    """Topic proportions for each document, rows summing to one."""
    return model.transform(matrix)


def topic_proportions(doc_topics: np.ndarray) -> list[float]:
    """Overall share of the corpus each topic accounts for."""
    return [round(float(v), 4) for v in doc_topics.mean(axis=0)]


def proportions_by_covariate(doc_topics: np.ndarray, covariate) -> dict[str, list[float]]:
    """Mean topic proportions within each level of a covariate.

    This is the post-hoc stand-in for STM's covariate-in-prior estimation: the model is
    fitted without the covariate, and prevalence is compared across its levels
    afterwards. It answers the same question the study's covariate figures answer, by a
    simpler route.
    """
    values = np.asarray(covariate)
    out: dict[str, list[float]] = {}
    for level in sorted(set(values.tolist())):
        mask = values == level
        if mask.sum() == 0:
            continue
        out[str(level)] = [round(float(v), 4) for v in doc_topics[mask].mean(axis=0)]
    return out
