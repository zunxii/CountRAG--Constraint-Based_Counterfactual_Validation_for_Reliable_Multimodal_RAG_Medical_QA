```mermaid
flowchart TB
    subgraph INPUT["📥 INPUT DATA"]
        A1[("CLIPSyntel Dataset<br/>667 medical cases<br/>Image + Patient Question<br/>+ Clinical Description")]
        A2["Data Split Module<br/>(split.py)"]
        A1 --> A2
        A2 --> A3["Train Split<br/>534 cases (80%)<br/>🔒 KB Building"]
        A2 --> A4["Eval Split<br/>133 cases (20%)<br/>🔒 Reserved Testing"]
    end

    subgraph LORA["🧬 PHASE 1: LoRA FINE-TUNING"]
        direction TB
        B1["Base Model<br/>BioMedCLIP<br/>(Microsoft 2023)"]
        B2["Vision Encoder<br/>ViT-B/16<br/>512-dim embeddings"]
        B3["Text Encoder<br/>PubMedBERT<br/>512-dim embeddings"]
        B1 --> B2
        B1 --> B3
        
        B4["LoRA Injection<br/>target_modules: all-linear<br/>r=8, α=16, dropout=0.05"]
        B2 --> B4
        B3 --> B4
        
        B5["Training Dataset<br/>Patient Questions ↔ Images<br/>Clinical Descriptions ↔ Images"]
        
        B6["Multi-Task Loss Function<br/>L = L_contrastive + 0.2·L_cycle + 0.05·L_align"]
        
        B7["L_contrastive (InfoNCE)<br/>-log[exp(sim(I,T)/τ) / Σ exp(sim(I,T_j)/τ)]<br/>τ = 0.07"]
        
        B8["L_cycle (Cycle Consistency)<br/>MSE(sim(Question, Image),<br/>sim(Clinical, Image))"]
        
        B9["L_align (Direct Alignment)<br/>CrossEntropy(sim(Question, Clinical)/τ)"]
        
        B6 --> B7
        B6 --> B8
        B6 --> B9
        
        B10["Training Loop<br/>Epochs: 15<br/>LR: 5e-5 (AdamW)<br/>Batch: 16<br/>Scheduler: CosineAnnealing"]
        
        B11["Early Stopping<br/>Patience: 3<br/>Monitor: Val Loss"]
        
        B12[("Trained LoRA Adapters<br/>outputs/models/trained_lora/<br/>adapter_model.bin<br/>~1.2 MB")]
        
        B4 --> B5
        B5 --> B6
        B10 --> B11
        B11 --> B12
    end

    subgraph FUSION["⚡ PHASE 2: FUSION TRAINING"]
        direction TB
        C1["Frozen Encoders<br/>LoRA-adapted BioMedCLIP<br/>No gradient updates"]
        
        C2["Adaptive Fusion Layer<br/>Input: img_emb ⊕ txt_emb<br/>Hidden: 256 dims"]
        
        C3["Gating Network<br/>g = σ(W₂·GELU(W₁·[img;txt]))"]
        
        C4["Residual Fusion<br/>fused = img + g·txt"]
        
        C5["Layer Normalization<br/>+ L2 Normalization"]
        
        C6["Training Data<br/>Question + Image pairs<br/>❌ NO Clinical Text"]
        
        C7["Contrastive Loss<br/>L = contrastive(fused, txt_emb)"]
        
        C8["Training Config<br/>Epochs: 10<br/>LR: 1e-4<br/>Batch: 16<br/>Weight Decay: 0.01"]
        
        C9["Validation Strategy<br/>90/10 split<br/>Early Stop: Patience=3"]
        
        C10[("Trained Fusion Model<br/>outputs/models/trained_fusion/<br/>fusion.pt<br/>~0.5 MB")]
        
        C1 --> C2
        C2 --> C3
        C3 --> C4
        C4 --> C5
        C5 --> C6
        C6 --> C7
        C7 --> C8
        C8 --> C9
        C9 --> C10
    end

    subgraph KB_CHOICE["🗄️ PHASE 3: KB ARCHITECTURE CHOICE"]
        direction LR
        D1{"KB Mode Selection"}
        D2["Flat Mode<br/>One entry per image<br/>534 entries"]
        D3["Concept Mode ⭐<br/>Grouped by diagnosis<br/>~18 concepts"]
        
        D1 -->|Traditional| D2
        D1 -->|Recommended| D3
    end

    subgraph FLAT["📊 FLAT KB PIPELINE"]
        direction TB
        E1["Load Train Split<br/>534 rows"]
        E2["For each row:<br/>• Load image<br/>• Load clinical text"]
        E3["Encode Image<br/>img_emb = encoder.encode_image(img)"]
        E4["Encode Text<br/>txt_emb = encoder.encode_text(clinical)"]
        E5["Fusion<br/>concept_emb = fusion(img_emb, txt_emb)"]
        E6["Store Entry<br/>• case_id<br/>• diagnosis_label<br/>• image_path<br/>• clinical_text<br/>• embedding_id"]
        E7["Build FAISS Index<br/>IndexFlatIP(512)<br/>Cosine similarity search"]
        E8["Normalize Embeddings<br/>L2_normalize(embeddings)"]
        E9[("Flat KB Output<br/>kb_final_v2/<br/>├─ embeddings.npy [534×512]<br/>├─ index.faiss<br/>├─ metadata.json<br/>└─ kb_config.json")]
        
        E1 --> E2
        E2 --> E3
        E3 --> E4
        E4 --> E5
        E5 --> E6
        E6 --> E7
        E7 --> E8
        E8 --> E9
    end

    subgraph CONCEPT["🎯 CONCEPT KB PIPELINE"]
        direction TB
        F1["Load Train Split<br/>534 rows"]
        F2["Group by Canonical Description<br/>Hash(clinical_text) → concept_id"]
        F3["Concept Grouping<br/>edema: 20 images<br/>cyanosis: 15 images<br/>..."]
        F4["For each concept:<br/>Encode all images"]
        F5["Image Aggregation<br/>v_proto = mean(img_embeddings)<br/>Options: mean/max/weighted"]
        F6["Encode Canonical Text<br/>v_text = encode(description)"]
        F7["Fuse Prototype + Text<br/>v_concept = fusion(v_proto, v_text)"]
        F8["Build Concept Metadata<br/>• concept_id<br/>• diagnosis_label<br/>• canonical_text<br/>• image_paths[]<br/>• num_images<br/>• image_variance"]
        F9["Create Multiple Indices<br/>• concept_index.faiss<br/>• text_index.faiss<br/>• image_index.faiss"]
        F10["Build Image→Concept Mapping<br/>For concept expansion"]
        F11[("Concept KB Output<br/>kb_final_concept/<br/>├─ concept_embeddings.npy<br/>├─ text_embeddings.npy<br/>├─ image_embeddings.npy<br/>├─ concept_index.faiss<br/>├─ metadata.json [18 concepts]<br/>└─ image_to_concept.json")]
        
        F1 --> F2
        F2 --> F3
        F3 --> F4
        F4 --> F5
        F5 --> F6
        F6 --> F7
        F7 --> F8
        F8 --> F9
        F9 --> F10
        F10 --> F11
    end

    subgraph VALIDATION["✅ KB VALIDATION"]
        direction TB
        G1["Load KB with KBRetriever"]
        G2["Verify Integrity<br/>• Index size matches metadata<br/>• No NaN embeddings<br/>• All files present"]
        G3["Statistics<br/>• Num entries/concepts<br/>• Embedding dimension<br/>• Mode (flat/concept)"]
        G4["Sample Query Test<br/>Encode sample → Search → Verify"]
        G5[("✓ KB Ready for Inference")]
        
        G1 --> G2
        G2 --> G3
        G3 --> G4
        G4 --> G5
    end

    subgraph OUTPUTS["📤 FINAL OUTPUTS"]
        H1[("Trained Models<br/>outputs/models/<br/>├─ trained_lora/<br/>│  ├─ adapter_model.bin<br/>│  ├─ adapter_config.json<br/>│  └─ training_history.json<br/>└─ trained_fusion/<br/>   ├─ fusion.pt<br/>   └─ training_history.json")]
        
        H2[("Knowledge Base<br/>outputs/kb/kb_final_concept/<br/>├─ concept_embeddings.npy<br/>├─ *_index.faiss<br/>├─ metadata.json<br/>└─ build_report.json")]
        
        H3[("Training Artifacts<br/>outputs/plots/<br/>├─ lora/<br/>│  ├─ lora_cycle_training_total.png<br/>│  └─ lora_cycle_training_cycle.png<br/>└─ fusion/<br/>   └─ fusion_training_curves.png")]
    end

    %% Connections between phases
    A3 --> B5
    B12 --> C1
    C10 --> D1
    D2 --> E1
    D3 --> F1
    E9 --> G1
    F11 --> G1
    G5 --> H1
    G5 --> H2
    B12 --> H1
    C10 --> H1
    G5 --> H3

    %% Styling
    classDef inputStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef loraStyle fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    classDef fusionStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    classDef kbStyle fill:#e8f5e9,stroke:#388e3c,stroke-width:3px
    classDef outputStyle fill:#fce4ec,stroke:#c2185b,stroke-width:3px
    classDef validationStyle fill:#e0f2f1,stroke:#00796b,stroke-width:3px
    
    class A1,A2,A3,A4 inputStyle
    class B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12 loraStyle
    class C1,C2,C3,C4,C5,C6,C7,C8,C9,C10 fusionStyle
    class D1,D2,D3,E1,E2,E3,E4,E5,E6,E7,E8,E9,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11 kbStyle
    class G1,G2,G3,G4,G5 validationStyle
    class H1,H2,H3 outputStyle