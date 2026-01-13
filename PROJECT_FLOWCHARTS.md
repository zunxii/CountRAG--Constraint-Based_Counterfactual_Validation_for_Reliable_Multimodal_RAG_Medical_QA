# Project Flowcharts

This document visualizes the **Research RAG** pipeline using flowcharts.

## 1. Complete Pipeline Overfiew

```mermaid
graph TD
    subgraph Data Prep
        RAW[Raw Data] --> CLEAN[Cleaned CSV]
        TRAIN_SPLIT[Train Split]
        EVAL_SPLIT[Eval Split]
    end

    subgraph Phase 1: Training
        CLEAN --> LORA[Train LoRA]
        LORA --> FUSION[Train Fusion]
        FUSION --> MODEL_OUT[Saved Models]
    end

    subgraph Phase 2: Knowledge Base
        MODEL_OUT --> KB_BUILD[Build KB]
        TRAIN_SPLIT --> KB_BUILD
        KB_BUILD --> CONCEPT_KB[Concept Knowledge Base]
        
        style CONCEPT_KB fill:#bbf,stroke:#333
    end

    subgraph Phase 3: Runtime
        USER[User Query] --> INFER[Inference Engine]
        CONCEPT_KB --> INFER
        MODEL_OUT --> INFER
        INFER --> RESPONSE[Final Answer]
    end
```

---

## 2. Concept-Based KB Construction (Detailed)

This explains how we turn raw data into a clean "Concept" index (`scripts/kb/build_kb.py`).

```mermaid
flowchart LR
    A[Raw Images (e.g. 50 photos of 'Rash')] --> B{Group by Description}
    B --> C[Concept Group: 'Rash Type A']
    
    subgraph Processing One Concept
        C --> D[Embed All 50 Images]
        D --> E[Aggregate (Mean)]
        E --> F[Prototypical Image Vector]
        
        C --> G[Canonical Text Description]
        G --> H[Text Encoder]
        H --> I[Text Vector]
        
        F --> J{Adaptive Fusion}
        I --> J
        J --> K[Final Concept Vector]
    end
    
    K --> L[(FAISS Index)]
```

---

## 3. Inference Logic

What happens when you run `python main.py infer`.

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Encoder
    participant Fusion
    participant Retriever
    participant KB

    User->>CLI: "Is this cancer?" + [Image]
    CLI->>Encoder: Encode Question & Image
    Encoder-->>CLI: Text Emb, Image Emb
    
    CLI->>Fusion: Fuse(Text Emb, Image Emb)
    Fusion-->>CLI: Query Vector
    
    CLI->>Retriever: Search(Query Vector)
    Retriever->>KB: Scan Index
    KB-->>Retriever: Top-k Concepts (Matches)
    Retriever-->>CLI: Return ["Melanoma", "Mole", ...]
    
    CLI->>User: Display Top Matches
```
