"""Text pre-processing for the employee review corpus.

The published study describes four steps: tokenisation and sentence segmentation,
stopword removal, lemmatisation and stemming, and the removal of duplicates and empty
reviews. This module implements each of them without external language models, so the
chain runs anywhere Python runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Common English stopwords, plus the platform boilerplate that appears in nearly every
# review and would otherwise dominate the bigram counts.
STOPWORDS: set[str] = set(
    """a about above after again against all am an and any are as at be because been before
    being below between both but by can cannot could did do does doing down during each few for
    from further had has have having he her here hers herself him himself his how i if in into is
    it its itself me more most my myself no nor not of off on once only or other ought our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why with would you your yours yourself yourselves
    get got also would like really lot much many one two will just even still always never every
    company companies job work working place employee employees review reviews""".split()
)

# Irregular forms the suffix rules would otherwise mangle.
IRREGULAR: dict[str, str] = {
    "was": "be", "were": "be", "been": "be", "being": "be", "is": "be", "are": "be",
    "had": "have", "has": "have", "having": "have", "did": "do", "does": "do",
    "went": "go", "gone": "go", "paid": "pay", "took": "take", "taken": "take",
    "made": "make", "left": "leave", "felt": "feel", "gave": "give", "given": "give",
    "people": "people", "children": "child", "men": "man", "women": "woman",
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
}

# Applied in order. The study lemmatises and then applies stemming, so the rules below
# combine both: inflectional endings first, then the derivational endings that a light
# stemmer removes. "developing" reduces to "develop", as the article's own example gives.
SUFFIX_RULES: tuple[tuple[str, str, int], ...] = (
    ("ies", "y", 4),
    ("sses", "ss", 5),
    ("ches", "ch", 5),
    ("shes", "sh", 5),
    ("ss", "ss", 2),
    ("s", "", 4),
    ("ing", "", 6),
    ("edly", "", 7),
    ("ed", "", 5),
    ("ement", "e", 7),
    ("ment", "", 6),
    ("ities", "ity", 6),
    ("ness", "", 6),
    ("ly", "", 5),
)

_TOKEN_RE = re.compile(r"[a-z][a-z'-]*")
_SENTENCE_RE = re.compile(r"[.!?;\n]+")
_ACCENTS = str.maketrans(
    "áàâäãåéèêëíìîïóòôöõúùûüñçÁÀÂÄÃÅÉÈÊËÍÌÎÏÓÒÔÖÕÚÙÛÜÑÇ",
    "aaaaaaeeeeiiiiooooouuuuncAAAAAAEEEEIIIIOOOOOUUUUNC",
)


@dataclass
class PreprocessConfig:
    """Knobs that change the shape of the token stream."""

    min_tokens: int = 2
    min_token_length: int = 3
    apply_stemming: bool = True
    extra_stopwords: set[str] = field(default_factory=set)


def normalise(text: str) -> str:
    """Lower-case, strip accents, punctuation and digits."""
    text = str(text).translate(_ACCENTS).lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z'\s.!?;\n-]", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def segment(text: str) -> list[str]:
    """Split normalised text into sentences."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def stem(token: str) -> str:
    """Reduce a token to its root form."""
    if token in IRREGULAR:
        return IRREGULAR[token]
    for suffix, replacement, min_length in SUFFIX_RULES:
        if token.endswith(suffix) and len(token) >= min_length:
            return token[: -len(suffix)] + replacement
    return token


def tokenise(text: str, config: PreprocessConfig | None = None) -> list[str]:
    """Normalise, tokenise, drop stopwords and reduce to root forms."""
    config = config or PreprocessConfig()
    stops = STOPWORDS | config.extra_stopwords
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(normalise(text)):
        token = raw.strip("'-")
        if len(token) < config.min_token_length or token in stops:
            continue
        root = stem(token) if config.apply_stemming else token
        if len(root) < config.min_token_length or root in stops:
            continue
        tokens.append(root)
    return tokens


def tokenise_sentences(text: str, config: PreprocessConfig | None = None) -> list[list[str]]:
    """Tokenise sentence by sentence.

    Bigrams are counted within sentences rather than across the whole review, so that the
    last word of one sentence and the first of the next are not treated as a pair.
    """
    config = config or PreprocessConfig()
    return [t for t in (tokenise(s, config) for s in segment(normalise(text))) if t]


def is_usable(text: str, config: PreprocessConfig | None = None) -> bool:
    """Whether a review survives the empty-review filter."""
    config = config or PreprocessConfig()
    return len(tokenise(text, config)) >= config.min_tokens


def deduplicate(texts: list[str]) -> list[int]:
    """Indices of the first occurrence of each distinct review.

    Duplicates are judged on the normalised text, so that two postings differing only in
    capitalisation or spacing count once.
    """
    seen: set[str] = set()
    keep: list[int] = []
    for i, text in enumerate(texts):
        key = " ".join(normalise(text).split())
        if key and key not in seen:
            seen.add(key)
            keep.append(i)
    return keep
