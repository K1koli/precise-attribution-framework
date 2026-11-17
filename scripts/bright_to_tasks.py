#!/usr/bin/env python3
"""把 BRIGHT 原始 parquet 转成本框架的任务格式（每题自带候选池 + gold_doc_ids）。

为什么需要候选池：BRIGHT 原始设定是“query + 全库”，而本框架的任务格式要求每题
自带 ``documents``。这里用 BM25 从全库检索 top-(pool-size) 并把该题所有 gold chunk
强行并入，得到一个可评测的候选池（做法与 BIRCO 的 rerank 设定一致）。

用法：
    python scripts/bright_to_tasks.py --raw-dir data/bright/raw --domain biology \
        --out configs/tasks/bright_biology_50.yaml --pool-size 50

依赖：pyarrow（读取 parquet）。生成的文件用 ``benchmark: bright_ndcg10`` 评测。
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """依赖轻量的 BM25（k1=1.2, b=0.75），只用于构造候选池。"""

    def __init__(self, docs: list[tuple[str, str]], *, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_ids = [d[0] for d in docs]
        self.tfs: list[Counter] = []
        self.doc_len: list[int] = []
        df: Counter = Counter()
        for _, text in docs:
            tokens = tokenize(text)
            tf = Counter(tokens)
            self.tfs.append(tf)
            self.doc_len.append(len(tokens))
            for term in tf:
                df[term] += 1
        n = len(docs)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
        self.postings: dict[str, list[int]] = defaultdict(list)
        for idx, tf in enumerate(self.tfs):
            for term in tf:
                self.postings[term].append(idx)

    def top_k(self, query: str, k: int) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx in self.postings.get(term, ()):  # type: ignore[union-attr]
                tf = self.tfs[idx][term]
                dl = self.doc_len[idx] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[idx] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.doc_ids[kv[0]]))[:k]
        return [(self.doc_ids[i], s) for i, s in ranked]


_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _clean(text: str) -> str:
    """去掉控制字符与孤立代理字符。

    BRIGHT 正文里有 \x80 这类控制字符（YAML 会拒绝），也有 U+D800–DFFF 的孤立
    代理字符（会让 json/utf-8 写入抛 UnicodeEncodeError）。两者都必须清掉。
    """
    text = _CTRL_RE.sub(" ", text)
    return "".join(ch for ch in text if not (0xD800 <= ord(ch) <= 0xDFFF))


def _title_of(doc_id: str) -> str:
    name = doc_id.rsplit("/", 1)[-1]
    name = re.sub(r"_\d+\.txt$", "", name)
    return name.replace("_", " ")[:120]


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("需要 pyarrow：pip install pyarrow（或用 --format jsonl 的已转换数据）") from exc
    return pq.read_table(path).to_pylist()


def convert(
    raw_dir: Path,
    domain: str,
    out: Path,
    *,
    pool_size: int = 50,
    limit: int | None = None,
    snippet_chars: int = 1200,
    min_gold: int = 1,
) -> dict[str, Any]:
    doc_rows = load_parquet_rows(raw_dir / "documents" / f"{domain}-00000-of-00001.parquet")
    ex_rows = load_parquet_rows(raw_dir / "examples" / f"{domain}-00000-of-00001.parquet")

    corpus = {str(row["id"]): str(row["content"]) for row in doc_rows}
    bm25 = BM25(list(corpus.items()))

    tasks: list[dict[str, Any]] = []
    skipped = {"no_gold_in_corpus": 0}
    pool_sizes: list[int] = []
    for row in ex_rows:
        gold = [str(g) for g in (row.get("gold_ids") or []) if str(g) in corpus]
        if len(gold) < min_gold:
            skipped["no_gold_in_corpus"] += 1
            continue
        query = str(row["query"])
        ranked = bm25.top_k(query, pool_size)
        pool = [doc_id for doc_id, _ in ranked]
        gold_set = set(gold)
        # 先保证 gold 在池内，再用 BM25 结果补满 pool_size（不能"追加后再截断"，否则
        # gold 会被切掉——这就是第一版 96/103 题金标缺失的原因）
        # 先把 BM25 结果里的 gold 全部剔除，再把"全部 gold"追加回来；
        # 只补 missing 会丢掉那些本来就在 BM25 top-k 里的 gold。
        non_gold = [doc_id for doc_id in pool if doc_id not in gold_set]
        pool = non_gold[: max(0, pool_size - len(gold))] + list(dict.fromkeys(gold))
        tasks.append({
            "id": f"bright-{domain}-{row.get('id')}",
            "question": query,
            "gold_doc_ids": gold,
            "documents": [
                {
                    "id": doc_id,
                    "title": _title_of(doc_id),
                    "text": _clean(corpus[doc_id][:snippet_chars]).strip(),
                }
                for doc_id in pool
            ],
        })
        pool_sizes.append(len(pool))
        if limit and len(tasks) >= limit:
            break

    payload = {
        "benchmark": "bright_ndcg10",
        "domain": domain,
        "pool_size": pool_size,
        "snippet_chars": snippet_chars,
        "notes": (
            "由 scripts/bright_to_tasks.py 生成：BM25 检索 top-pool_size 候选并并入全部 gold chunk。"
            "评测用 nDCG@10，只对 QueryUnderstanding / EvidenceRetrieval 做归因。"
        ),
        "stats": {
            "queries": len(tasks),
            "corpus_documents": len(corpus),
            "pool_size_min": min(pool_sizes) if pool_sizes else 0,
            "pool_size_max": max(pool_sizes) if pool_sizes else 0,
            "gold_total": sum(len(t["gold_doc_ids"]) for t in tasks),
            "skipped": skipped,
        },
        "tasks": tasks,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # 必须 ensure_ascii=False + utf-8：ensure_ascii=True 会把表情写成 \ud83d\ude42
    # 这类代理对转义，YAML 解析回来就变成两个孤立代理，写报告时抛 UnicodeEncodeError。
    # 控制字符与孤立代理已在 _clean 里清掉，非 BMP 字符（emoji）保持原样即可。
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload["stats"]


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BRIGHT → 本框架任务格式")
    ap.add_argument("--raw-dir", required=True, help="包含 documents/ 与 examples/ 的原始目录")
    ap.add_argument("--domain", default="biology", help="biology / economics / robotics")
    ap.add_argument("--out", required=True, help="输出文件（建议 .yaml 或 .json，两者都能被 yaml.safe_load 读取）")
    ap.add_argument("--pool-size", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None, help="只转换前 N 个 query")
    ap.add_argument("--snippet-chars", type=int, default=1200)
    ap.add_argument("--min-gold", type=int, default=1)
    args = ap.parse_args(argv)

    stats = convert(
        Path(args.raw_dir), args.domain, Path(args.out),
        pool_size=args.pool_size, limit=args.limit,
        snippet_chars=args.snippet_chars, min_gold=args.min_gold,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
