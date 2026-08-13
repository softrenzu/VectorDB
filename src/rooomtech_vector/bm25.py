from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+", re.UNICODE)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def tokenize(text: str) -> list[str]:
    """Dependency-free tokenizer; CJK runs are emitted as character bigrams."""
    out: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if _CJK_RE.search(token):
            chars = list(token)
            if len(chars) == 1:
                out.append(token)
            else:
                out.extend("".join(chars[i : i + 2]) for i in range(len(chars) - 1))
        else:
            out.append(token)
    return out


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not documents:
        return []
    tokenized = [tokenize(doc) for doc in documents]
    query_terms = tokenize(query)
    if not query_terms:
        return [0.0] * len(documents)

    n_docs = len(documents)
    avgdl = sum(len(doc) for doc in tokenized) / max(n_docs, 1)
    doc_freq: Counter[str] = Counter()
    for doc in tokenized:
        doc_freq.update(set(doc))

    result: list[float] = []
    for doc in tokenized:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in query_terms:
            if tf[term] == 0:
                continue
            df = doc_freq[term]
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf[term] + k1 * (1.0 - b + b * dl / max(avgdl, 1e-9))
            score += idf * (tf[term] * (k1 + 1.0)) / denom
        result.append(float(score))
    return result
