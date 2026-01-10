# CLI Commands (Copy–Paste Reference)

## Training
- `python main.py train-lora`  
  Trains LoRA adapters for the encoder models.

- `python main.py train-fusion`  
  Trains the multimodal fusion module.

## Knowledge Base (KB)
- `python main.py build-kb`  
  Builds the full production knowledge base and FAISS index.

- `python main.py build-kb-smoke`  
  Builds a small smoke-test KB for quick validation.

## Mean aggregation (default, robust)
`python scripts/kb/build_kb.py --mode concept --aggregation mean`

## Max aggregation (emphasizes extremes)
`python scripts/kb/build_kb.py --mode concept --aggregation max`

## Weighted aggregation (by embedding norm)
`python scripts/kb/build_kb.py --mode concept --aggregation weighted`

## Pipeline
- `python main.py pipeline`  
  Runs the full configurable pipeline.

- `python main.py pipeline --config path/to/config.yaml`  
  Runs pipeline using a specific YAML configuration.

- `python main.py pipeline --skip-training`  
  Runs pipeline while skipping all training stages.

- `python main.py pipeline --stages train kb eval`  
  Runs only selected pipeline stages.

## Inference
- `python main.py infer --query-text "your query"`  
  Runs single-query inference using existing models and KB.

- `python main.py infer --query-text "your query" --query-image path/to/image.jpg`  
  Runs multimodal inference with text + image input.

- `python main.py infer --query-text "your query" --kb-dir outputs/kb/kb_final_v2 --top-k 5`  
  Runs inference with a custom KB path and top-k retrieval.

- `python main.py reason --query-text "your query" --query-image path/to/image.jpg`  
  Runs counterfactual reasoning and stability analysis.

- `python main.py reason-full`  
  Runs the full counterfactual inference pipeline with explanation layer.

## Evaluation
- `python main.py evaluate-retrieval`  
  Runs retrieval evaluation (legacy entrypoint).

- `python main.py eval-retrieval --kb-dir outputs/kb/kb_final_v2 --output-dir outputs/evaluation/retrieval`  
  Runs detailed retrieval evaluation and saves results.

- `python main.py eval-encoders --output-dir outputs/evaluation/encoders`  
  Evaluates encoder and fusion representations.

- `python main.py eval-counterfactual --kb-dir outputs/kb/kb_final_v2 --num-samples 50 --output-dir outputs/evaluation/counterfactual`  
  Runs quantitative counterfactual stability evaluation.

- `python main.py eval-lora --output-dir outputs/evaluation/lora`  
  Evaluates the impact of LoRA fine-tuning.

## Testing
- `python main.py test`  
  Runs the full test suite.

- `python main.py test -v`  
  Runs tests in verbose mode.

- `python main.py test --module <name>`  
  Runs a specific test module (`tests/test_<name>.py`).
