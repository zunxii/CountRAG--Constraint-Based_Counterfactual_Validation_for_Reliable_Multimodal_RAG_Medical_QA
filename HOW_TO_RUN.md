# How to Run Research RAG Pipeline

## Prerequisites

Ensure you have Python 3.8+ installed.

## Environment Setup (MacOS)

It is recommended to use a virtual environment to manage dependencies.

1.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    ```

2.  **Activate the Virtual Environment:**
    ```bash
    source venv/bin/activate
    ```

## Installation

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## CLI Commands

The project is controlled via `main.py`. Below are the primary commands.

### 1. Training
Train the models before building the Knowledge Base.

-   **Train LoRA Adapters:**
    ```bash
    python main.py train-lora
    ```
-   **Train Fusion Module:**
    ```bash
    python main.py train-fusion
    ```

### 2. Building Knowledge Base (KB)
Create the searchable index from your data.

-   **Build Full KB:**
    ```bash
    python main.py build-kb
    ```
-   **Build Smoke Test KB (Quick Validation):**
    ```bash
    python main.py build-kb-smoke
    ```

### 3. Pipeline (Automated Workflow)
Run multiple stages (Training -> KB -> Eval) in one go.

-   **Run Full Pipeline:**
    ```bash
    python main.py pipeline
    ```
-   **Skip Training (Run KB + Eval only):**
    ```bash
    python main.py pipeline --skip-training
    ```
-   **Run Specific Stages:**
    ```bash
    python main.py pipeline --stages train eval
    ```

### 4. Inference
Query the system.

-   **Single Query (Text):**
    ```bash
    python main.py infer --query-text "What are the symptoms of X?"
    ```
-   **Multimodal Query (Text + Image):**
    ```bash
    python main.py infer --query-text "Analyze this scan" --query-image "path/to/scan.jpg"
    ```
-   **Counterfactual Reasoning:**
    ```bash
    python main.py reason --query-text "What if..." --query-image "path/to/img.jpg"
    ```

### 5. Evaluation
Assess model performance.

-   **Retrieval Evaluation:**
    ```bash
    python main.py eval-retrieval
    ```
-   **Encoder Evaluation:**
    ```bash
    python main.py eval-encoders
    ```
-   **Full Evaluations:**
    See `EVALUATION_GUIDE.md` for detailed metrics and advanced usage.

### 6. Testing
Run the codebase test suite.

-   **Run All Tests:**
    ```bash
    python main.py test
    ```
