# CountRAG

CountRAG is a multimodal medical RAG research project that combines LoRA-based encoder adaptation, adaptive fusion, concept-based knowledge base construction, and counterfactual reasoning for retrieval and validation.

The repository is organized around a single CLI entry point, `main.py`, which dispatches commands to specialized scripts under `scripts/`, while reusable model and retrieval logic lives under `core/`. The repo also includes project notes and run guides such as `HOW_TO_RUN.md`, `EVALUATION_GUIDE.md`, `PROJECT_FLOW.md`, `PROJECT_COMPREHENSIVE_GUIDE.md`, `commands.md`, and `kb.md`. citeturn809312view0turn431453view1turn431453view3turn431453view0turn431453view2

## What this project does

The pipeline is built around these stages:

1. **Train LoRA adapters** for BioMedCLIP-style multimodal encoders.
2. **Train an adaptive fusion module** for text-image representation mixing.
3. **Build a knowledge base** with either a flat index or a concept-level index.
4. **Run inference** on text-only or text+image queries.
5. **Run counterfactual reasoning** to analyze modality sensitivity and stability.
6. **Evaluate** retrieval, encoder behavior, LoRA impact, and counterfactual robustness. citeturn616262view2turn916758view1turn431453view0

## Repository structure

```text
.
├── configs/
├── core/
│   ├── embeddings/
│   ├── fusion/
│   ├── kb/
│   ├── reasoning/
│   └── retrieval/
├── data/
├── outputs/
├── scripts/
│   ├── evaluation/
│   ├── inference/
│   ├── kb/
│   ├── pipeline/
│   ├── training/
│   └── utils/
├── tests/
├── EVALUATION_GUIDE.md
├── HOW_TO_RUN.md
├── PROJECT_COMPREHENSIVE_GUIDE.md
├── PROJECT_FLOW.md
├── PROJECT_FLOWCHARTS.md
├── commands.md
├── generate_all_plots.py
├── kb.md
├── main.py
├── requirements.txt
├── setup.py
├── split.py
└── .gitignore
```

This matches the GitHub tree and the nested module layout shown in the repo. The `core/` package contains `embeddings`, `fusion`, `kb`, `reasoning`, and `retrieval`, while `scripts/` contains `evaluation`, `inference`, `kb`, `pipeline`, `training`, and `utils`. citeturn809312view0turn759919view0turn511845view0turn650401view0turn650401view1turn650401view2turn650401view3turn650401view4

## Main CLI commands

The repository uses `main.py` as the entry point. The documented commands include:

- `python main.py train-lora`
- `python main.py train-fusion`
- `python main.py build-kb`
- `python main.py build-kb-smoke`
- `python main.py build-kb-dry`
- `python main.py pipeline`
- `python main.py infer --query-text "..."`
- `python main.py infer --query-text "..." --query-image path/to/image.jpg`
- `python main.py reason --query-text "..." --query-image path/to/image.jpg`
- `python main.py reason-full`
- `python main.py eval-retrieval`
- `python main.py eval-encoders`
- `python main.py eval-counterfactual`
- `python main.py eval-lora`
- `python main.py test` citeturn956269view1turn956269view2turn956269view3turn956269view4turn956269view5turn431453view0turn916758view1

## Core design

### 1. LoRA adaptation
The docs describe LoRA-based adaptation of BioMedCLIP with low-rank matrices and a multi-task loss made of contrastive alignment, cycle consistency, and direct text alignment. The KB guide lists the training setup as `r=8`, `α=16`, `dropout=0.05`, with a total loss of `L = L_contrastive + 0.2·L_cycle + 0.05·L_align`. citeturn431453view2turn212717view1

### 2. Adaptive fusion
The fusion module uses a gated residual design:
`g = σ(W₂·GELU(W₁·[img;txt]))`
and
`fused = img + g·txt`.
The repo docs say this module is trained after the encoders are frozen. citeturn431453view2turn212717view1

### 3. Knowledge base building
The knowledge base supports:
- **Flat mode**: one entry per image
- **Concept mode**: grouped by diagnosis, recommended in the docs

The KB guide says concept mode groups cases into about 18 concepts and builds FAISS indexes for retrieval. citeturn431453view2

### 4. Counterfactual reasoning
The reasoning layer validates retrieval under modality removal and noise perturbation, then measures stability with JS divergence and related statistics. The evaluation guide also includes a counterfactual stability evaluation command. citeturn916758view1turn616262view4

## Dataset notes from the repo docs

The `kb.md` document describes the dataset split used in this codebase as:

- **667 total medical cases**
- **534 train**
- **133 eval**

It also shows outputs being written to `outputs/models/` and `outputs/kb/`. citeturn431453view2turn916758view0

## Evaluation

The evaluation guide reports these outputs and metrics:

- Retrieval: `R@K`, `Precision@K`, `MRR`, `MAP`, `NDCG`
- Encoder evaluation: normalization, determinism, dimensions, gating behavior, modality alignment
- Counterfactual evaluation: modality removal, noise robustness, JS divergence, per-diagnosis stability
- LoRA evaluation: base vs LoRA embedding comparison and alignment improvement citeturn916758view1

## How to run

The repo’s `HOW_TO_RUN.md` says the project is controlled through `main.py` and can be run with a virtual environment plus `pip install -r requirements.txt`. citeturn212717view2turn616262view1

## Reported results

The accompanying paper reports:

- **Top-1 retrieval accuracy:** 78.4%
- **MRR:** 0.808
- **Overall safety violations:** 49.6% → 18.7%
- **Modality-conflict error reduction:** 74.3% citeturn0file0L14-L18turn0file0L880-L898

## License

No license file is visible in the repository root listing that I could verify from GitHub’s tree view. citeturn809312view0

---

If you want, I can turn this into a cleaner GitHub-style README with badges, installation steps, usage examples, and a polished project description matching the repo exactly.
