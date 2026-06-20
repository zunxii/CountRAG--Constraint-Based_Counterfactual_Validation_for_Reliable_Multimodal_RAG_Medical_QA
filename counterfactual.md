┌──────────────────────────────────────────────────────────────┐
│              COUNTERFACTUAL RAG INFERENCE PIPELINE           │
└──────────────────────────────────────────────────────────────┘

INPUT: Single Query (text + image)

[Layer 1] Standard Retrieval
    Query → Fusion → Embedding → FAISS → Top-K

[Layer 2] Counterfactual Perturbations
    ├─ No-Text Variant → Top-K₁
    ├─ No-Image Variant → Top-K₂  
    └─ Noisy Variant → Top-K₃

[Layer 3] Stability Analysis
    ├─ JS Divergence Computation
    ├─ Cluster Distribution Comparison
    └─ Robustness Classification (high/medium/low)

[Layer 4] Constraints Extraction
    ├─ Modality Consistency (cosine_sim)
    ├─ Boundary Analysis (margin)
    ├─ Evidence Diversity (unique_cases)
    └─ OOD Detection (distance_check)

[Layer 5] Hypothesis Scoring
    For each diagnosis:
        - Retention Score (appears in X/4 variants)
        - Modality Dependency (multimodal/image/text/unstable)
        - Final Score (retention × base_support × modality_bonus)

[Layer 6] Explanation (Optional)
    Structured Output → Gemini → Natural Language

OUTPUT: Ranked Diagnoses + Stability Metrics + Constraints


┌──────────────────────────────────────────────────────────────┐
│           EVALUATION: Testing Core Architecture              │
└──────────────────────────────────────────────────────────────┘

INPUT: 200 Reserved Test Queries

[A] Retrieval Metrics (Standard IR)
    - Recall@K, Precision@K, MRR, MAP, NDCG
    - Per-diagnosis breakdown
    - Mode comparison (text/image/fusion)

[B] Stability Evaluation (Counterfactual Quality)
    Run Core Architecture on ALL test queries:
        - Average JS Divergence per modality
        - Robustness distribution (high/medium/low counts)
        - Per-diagnosis stability profiles
        - Modality effect statistical tests (t-tests)

[C] Encoder/Fusion Quality
    - Embedding normalization checks
    - Fusion gate statistics
    - Modality alignment tests

[D] LoRA Fine-tuning Impact
    - Base vs LoRA embedding comparison
    - Alignment improvement metrics

OUTPUT: Research Paper Metrics + Statistical Significance