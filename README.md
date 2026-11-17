# Getting Started

## Installation

需要 Python 3.12 或 3.13。

```bash
git clone --branch xiaoyan/harness-evolution https://github.com/K1koli/precise-attribution-framework.git
cd precise-attribution-framework

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install huggingface_hub pyarrow
```

## Data Preparation

### 下载数据

下载到 `data/bright/raw/`，保留官方目录结构：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="xlangai/BRIGHT",
    repo_type="dataset",
    local_dir="data/bright/raw",
)
PY
```

### 转换数据

修改 `SUBSET` 选择子集，例如 `biology`、`economics` 或 `robotics`：

```bash
SUBSET=biology
SEED=42
SPLIT_RATIO=(4 3 3)

python scripts/prepare_bright_full.py \
  --raw-dir data/bright/raw \
  --domain "$SUBSET" \
  --seed "$SEED" \
  --split-ratio "${SPLIT_RATIO[@]}" \
  --out "data/bright/processed/$SUBSET"
```

`SPLIT_RATIO=(4 3 3)` 对应 `[train, dev, test]`，seed 默认为 42。prepare 时完成划分，后续直接读取对应文件。Biology 共 103 题，划分为 41/31/31 题。

输出目录须为空。转换后：

```text
data/bright/
├── raw/                       # 官方原始数据
└── processed/biology/
    ├── train.json             # 训练问题及标注
    ├── dev.json               # 开发评估问题及标注
    ├── test.json              # 最终评估问题及标注
    └── corpus.json            # 该子集的完整检索语料
```

## Baseline Evaluation

在前面 prepare 好的 `data/bright/processed/<subset>/test.json` 上评估初始 Harness：使用原始 query 检索同目录下的 `corpus.json`，通过 BM25 返回 top-10，计算 nDCG@10。直接使用已保存的 test 划分，不再重新 split。

```bash
SUBSET=biology

python scripts/evaluate_bright_baseline.py \
  --subset "$SUBSET" \
  --output "data/derived/${SUBSET}-bm25-test"
```

`SUBSET` 与数据准备时保持一致。结果保存到输出目录中的 `test_report.json`，包含逐题检索结果和平均 nDCG@10；输出目录不能已存在。
