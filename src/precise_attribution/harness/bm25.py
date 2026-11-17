"""Full-corpus BM25 with a cached index and deterministic ranking."""
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, documents):
        self.documents = documents
        self.lengths = []
        self.postings = defaultdict(list)
        for i, doc in enumerate(documents):
            counts = Counter(tokens(doc["text"]))
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings[term].append((i, count))
        self.avgdl = sum(self.lengths) / max(1, len(documents)) or 1

    def search(self, queries, k, excluded=()):
        scores = defaultdict(float)
        n = len(self.documents)
        for query in queries:
            for term in set(tokens(query)):
                posting = self.postings.get(term, ())
                idf = math.log(1 + (n - len(posting) + .5) / (len(posting) + .5))
                for i, tf in posting:
                    scores[i] += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * self.lengths[i] / self.avgdl))
        excluded = set(excluded)
        ranked = sorted((i for i in range(n) if self.documents[i]["id"] not in excluded),
                        key=lambda i: (-scores.get(i, 0), self.documents[i]["id"]))[:k]
        return [{**self.documents[i], "retrieval_score": scores.get(i, 0)} for i in ranked]


@lru_cache(maxsize=2)
def corpus_index(path, sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Corpus checksum mismatch")
    return BM25Index(json.loads(raw))


def bm25_retrieve(inputs):
    if inputs.get("corpus_file"):
        index = corpus_index(inputs["corpus_file"], inputs["corpus_sha256"])
    else:
        index = BM25Index(inputs.get("documents", []))
    return {"documents": index.search(inputs.get("queries", []), int(inputs.get("k", 10)), inputs.get("excluded_ids", []))}


class IdentityQuery:
    def run(self, inputs):
        return {"queries": [str(inputs["question"])]}


class BM25Retrieval:
    def run(self, inputs):
        return bm25_retrieve(inputs)
