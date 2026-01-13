# Research RAG: Comprehensive Technical Guide

This document provides a **deep technical analysis** of the Research RAG project. It details the mathematical formulations, architectural decisions, and the "why" behind every component.

---

## 1. Domain Adaptation (LoRA Training)

The first phase (`train-lora.py`) adapts the general-purpose `BioMedCLIP` model to the specific "Patient Question <-> Clinical Description" domain.

### 1.1 Model Architecture: LoRA (Low-Rank Adaptation)
Instead of fine-tuning the full weight matrix $W \in \mathbb{R}^{d \times d}$ (which is computationally expensive and prone to catastrophic forgetting), we inject low-rank decomposition matrices $A$ and $B$.

$$ W' = W + \Delta W = W + BA $$

Where:
-   $B \in \mathbb{R}^{d \times r}$ (initialized to 0)
-   $A \in \mathbb{R}^{r \times d}$ (Gaussian initialization)
-   $r \ll d$ is the rank (we use $r=8$).

**Why?**
This allows us to update only ~0.5% of the parameters while retaining the strong pre-trained knowledge of BioMedCLIP. The `lora_alpha=16` hyperparameter scales the update $\Delta W$ by $\frac{\alpha}{r}$.

### 1.2 Training Objective: Multi-Task Loss
We train using a weighted sum of three loss components:

$$ L_{total} = L_{contrastive} + \lambda_{cycle} L_{cycle} + \lambda_{align} L_{align} $$

#### A. Contrastive Loss (InfoNCE)
Standard CLIP loss. Maximizes the dot product of matching image-text pairs while minimizing non-matches in the batch.

$$ L_{i \to t} = -\log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)} $$

Where $\text{sim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$ (Cosine Similarity) and $\tau$ is temperature ($0.07$).

#### B. Cycle Consistency Loss ($L_{cycle}$)
**Goal:** Ensure the model's understanding is consistent. If a "Patient Question" implies an "Image", that "Image" should implies the corresponding "Clinical Description".

Formula (from `core/embeddings/cycle_consistency.py`):
$$ L_{cycle} = \frac{1}{N} \sum_{i=1}^{N} \left( \text{sim}(Q_i, I_i) - \text{sim}(C_i, I_i) \right)^2 $$

Where $Q$ is Question embedding, $C$ is Clinical embedding, $I$ is Image embedding. We use MSE to force the *Question-Image* relationship to mimic the *Clinical-Image* relationship.

#### C. Alignment Loss ($L_{align}$)
**Goal:** Directly force the embedding space of "Layman Questions" to align with "Technical Descriptions".

$$ L_{align} = \text{CrossEntropy}(\text{sim}(Q, C) / \tau) $$

---

## 2. Multimodal Fusion (`train-fusion.py`)

Once the encoders are adapted, we freeze them and train a lightweight **Adaptive Gated Fusion** module (`core/fusion/adaptive_fusion.py`).

### 2.1 The Problem
Simply averaging text and image vectors ($v = \frac{v_{img} + v_{text}}{2}$) is suboptimal because sometimes the text is irrelevant (vague question) or the image is poor (blurry).

### 2.2 The Solution: Gated Residual Fusion
We learn a gating scalar $g \in [0, 1]$ that decides how much to let the text influence the image backbone.

1.  **Concatenate**: $x = [v_{img} ; v_{text}]$
2.  **Compute Gate**:
    $$ g = \sigma(W_2 \cdot \text{GELU}(W_1 \cdot x)) $$
    where $\sigma$ is Sigmoid.
3.  **Fuse (Residual)**:
    $$ v_{fused} = \text{LayerNorm}(v_{img} + g \cdot v_{text}) $$

**Why?**
-   **Residual connection ($v_{img} + ...$)**: Ensures that if the text is garbage ($g \approx 0$), we still preserve the original image features perfectly.
-   **Gating**: The model dynamically learns context. If the text says "red rash" and the image shows a red rash, $g$ increases to reinforce the signal.

---

## 3. Knowledge Base Construction (`build_kb.py`)

We use a **Concept-Based** approach rather than a standard flat index.

### 3.1 "Flat" vs. "Concept"
-   **Flat**: Index every image separately.
    -   *Problem*: A common disease with 1,000 images will dominate the search results (flooding), pushing rare diseases out of the top-k.
-   **Concept**: Group images by diagnosis/description.

### 3.2 Concept Vector Formation
For a concept $C$ (e.g., "Swollen Tonsils") with $k$ images $\{I_1, ..., I_k\}$:

1.  **Image Aggregation**: We compute the **Centroid** of the image cluster.
    $$ v_{prototype} = \frac{1}{k} \sum_{j=1}^{k} \text{Encode}(I_j) $$
    *Rationale*: By the Law of Large Numbers, averaging vectors reduces random noise (lighting, angle artifacts) and reinforces the core visual features of the disease.

2.  **Text Encoding**:
    $$ v_{text} = \text{Encode}(T_{canonical}) $$

3.  **Final Concept Vector**: Apply the trained fusion.
    $$ v_{concept} = \text{Fusion}(v_{prototype}, v_{text}) $$

This single vector $v_{concept}$ represents the *entire* medical condition in the vector space.

---

## 4. Inference & Retrieval

### 4.1 Search Mechanism
We use **FAISS** (Facebook AI Similarity Search) with `IndexFlatIP` (Inner Product).
Since all our vectors are L2-normalized ($\|v\|=1$), Inner Product is mathematically equivalent to Cosine Similarity.

$$ \text{sim}(q, d) = \sum q_i d_i $$

### 4.2 Algorithm
1.  **Encode**: User Query $Q_{text}$ and $Q_{img}$ are encoded via LoRA.
2.  **Fuse**: $Q_{final} = \text{Fusion}(Q_{img}, Q_{text})$.
3.  **Retrieve**: Find Top-K concepts maximizing $Q_{final} \cdot v_{concept}$.
4.  (Optional) **Expand**: Once a concept is found (e.g., "Eczema"), we can look up the specific images belonging to that concept to show the user.

---

## 5. Summary of Formulas

| Component | Formula | Purpose |
| :--- | :--- | :--- |
| **LoRA** | $W + BA$ | Efficient fine-tuning |
| **InfoNCE** | $-\log \frac{e^{s_{pos}/\tau}}{\sum e^{s_j/\tau}}$ | Contrastive alignment |
| **Cycle Loss** | $\| (Q \cdot I) - (C \cdot I) \|^2$ | Semantic consistency |
| **Fusion** | $I + \sigma(mlp([I;T])) \cdot T$ | Adaptive modal mixing |
| **Concept Agg** | $\frac{1}{N} \sum I_j$ | Noise reduction |

