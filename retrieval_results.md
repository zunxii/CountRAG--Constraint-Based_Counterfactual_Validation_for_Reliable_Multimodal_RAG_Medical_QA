
# Retrieval Performance Evaluation

This table compares the performance of different retrieval modes with and without LoRA fine-tuning.

## Comparative Results

| Method | LoRA Enabled | R@1 | R@5 | R@10 | MRR | MAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fusion** | ✅ | **0.7836** | 0.8399 | 0.8705 | **0.8082** | **0.8110** |
| Image | ✅ | 0.7479 | **0.8467** | **0.8978** | 0.7835 | 0.7671 |
| Text | ✅ | 0.6065 | 0.6422 | 0.6695 | 0.6235 | 0.6302 |
| | | | | | | |
| **Fusion** | ❌ | 0.6746 | 0.7394 | 0.7632 | 0.7041 | 0.7064 |
| Image | ❌ | 0.5741 | 0.6763 | 0.7530 | 0.6394 | 0.6712 |
| Text | ❌ | 0.3646 | 0.4497 | 0.4923 | 0.4105 | 0.4312 |

## Key Findings

1.  **LoRA Impact**: Enabling LoRA significantly improves performance across all modalities. 
    *   Fusion R@1 improved from **0.6746** to **0.7836** (+16%).
    *   Text R@1 improved drastically from **0.3646** to **0.6065** (+66%), showing that LoRA effectively adapted the text encoder to the medical domain.
    
2.  **Fusion Analysis**: Fusion consistently outperforms single modalities in Top-1 accuracy (R@1) and MRR, confirming the benefit of multimodal retrieval.

3.  **Modality Comparison**: Image-based retrieval generally performs better than Text-based retrieval in this dataset, but Fusion combines the strengths of both (especially with LoRA) to achieve the highest R@1 and MRR.
