
# Single-Modality vs. Neutral-Fusion Retrieval Performance

This table compares standard single-modality retrieval against "Neutral Fusion" approaches.

*   **Standard Text**: Query using only text embedding.
*   **Neutral Text**: Query using Fusion model with (Actual Text + Mean Image).
*   **Standard Image**: Query using only image embedding.
*   **Neutral Image**: Query using Fusion model with (Actual Image + Mean Text).

## Comparative Results

| Retrieval Mode | LoRA Enabled | R@1 | R@5 | R@10 | MRR | MAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fusion (Full)** | ✅ | **0.7836** | 0.8399 | 0.8705 | **0.8082** | **0.8110** |
| | | | | | | |
| **Neutral Image** (Image + Mean Text) | ✅ | **0.7734** | 0.8245 | 0.8637 | **0.7991** | 0.7636 |
| Standard Image | ✅ | 0.7479 | **0.8467** | **0.8978** | 0.7835 | **0.7671** |
| | | | | | | |
| Standard Text | ✅ | **0.6065** | **0.6422** | **0.6695** | **0.6235** | **0.6302** |
| **Neutral Text** (Text + Mean Image) | ✅ | 0.5945 | 0.6337 | 0.6678 | 0.6175 | 0.6203 |
| | | | | | | |
| **Fusion (Full)** | ❌ | 0.6746 | 0.7394 | 0.7632 | 0.7041 | 0.7064 |
| | | | | | | |
| **Neutral Image** (Image + Mean Text) | ❌ | **0.5792** | 0.6457 | 0.6593 | 0.6145 | 0.6266 |
| Standard Image | ❌ | 0.5741 | **0.6763** | **0.7530** | **0.6394** | **0.6712** |
| | | | | | | |
| **Neutral Text** (Text + Mean Image) | ❌ | **0.3816** | **0.4770** | **0.5298** | **0.4300** | **0.4425** |
| Standard Text | ❌ | 0.3646 | 0.4497 | 0.4923 | 0.4105 | 0.4312 |

## Analysis

1.  **Neutral Image Strategy**: Replacing the missing text modality with an "average text" vector (`Neutral Image`) works surprisingly well.
    *   **With LoRA**: It achieves **0.7734 R@1**, outperforming valid standard image retrieval (0.7479) and coming very close to the full fusion model (0.7836). This suggests the fusion model learns to leverage the image stream effectively even when the text stream is non-informative (average), effectively acting as a superior image encoder.
    
2.  **Neutral Text Strategy**:
    *   **Without LoRA**: The `Neutral Text` approach (Text + Mean Image) improves over standard Text retrieval (R@1: 0.3816 vs 0.3646). This indicates the fusion layers provide some benefit even with static visual context.
    *   **With LoRA**: Standard Text retrieval slightly outperforms Neutral Text (R@1: 0.6065 vs 0.5945). The LoRA-adapted text encoder is already very strong, and adding a mean image embedding might introduce minor noise or shift the distribution slightly away from what the fusion head expects for pure text signals.

3.  **Overall**: The "Neutral Fusion" strategy is a viable way to handle missing modalities, often matching or outperforming single-modality baselines by leveraging the trained fusion layers.
