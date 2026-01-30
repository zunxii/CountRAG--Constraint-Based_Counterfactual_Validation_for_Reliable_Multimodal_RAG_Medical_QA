import numpy as np
import torch
from .perturbations import remove_text, remove_image, add_noise
from .distribution import cluster_distribution
from .stability_metrics import stability_report
from ...constraints.extractor import ConstraintExtractor

class StabilityRunner:
    def __init__(self, retriever, fusion):
        self.retriever = retriever
        self.fusion = fusion.eval()
        self.extractor = ConstraintExtractor()

    def _run(self, img_emb, txt_emb):
        with torch.no_grad():
            fused = self.fusion(img_emb, txt_emb)
        retrieved = self.retriever.retrieve(fused)
        
        # Basic distribution stats
        stats = cluster_distribution(retrieved)
        
        return stats, retrieved, fused

    def _calculate_centroid_distances(self, fused_emb, retrieved):
        """
        Calculate cosine distance from query to centroids of retrieved clusters.
        """
        if not retrieved:
            return {}

        query_vec = fused_emb.cpu().numpy().flatten()
        # Normalize query if not already
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        # Group embeddings by label
        clusters = {}
        for r in retrieved:
            label = r["diagnosis_label"]
            emb = np.array(r["embedding"])
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(emb)
            
        distances = {}
        for label, embs in clusters.items():
            # Compute centroid
            embs_matrix = np.array(embs)
            centroid = np.mean(embs_matrix, axis=0)
            centroid = centroid / np.linalg.norm(centroid) # Normalize centroid
            
            # Cosine distance = 1 - cosine_similarity
            sim = np.dot(query_vec, centroid)
            dist = 1.0 - sim
            distances[label] = float(dist)
            
        return distances

    def run(self, img_emb, txt_emb):
        baseline, baseline_retrieved, baseline_fused = self._run(img_emb, txt_emb)

        i, t = remove_text(img_emb, txt_emb)
        no_text, _, _ = self._run(i, t)

        i, t = remove_image(img_emb, txt_emb)
        no_image, _, _ = self._run(i, t)

        noisy, _, _ = self._run(add_noise(img_emb), txt_emb)

        stability = stability_report(
            baseline,
            {
                "no_text": no_text,
                "no_image": no_image,
                "noisy": noisy,
            }
        )

        # Prepare arguments for ConstraintExtractor
        centroid_distances = self._calculate_centroid_distances(baseline_fused, baseline_retrieved)
        
        # Query Distance: Distance to top-1 result (1 - score)
        # Assuming score is Cosine Similarity
        top_score = baseline_retrieved[0]["score"] if baseline_retrieved else 0.0
        query_distance = 1.0 - top_score
        
        # Percentile 95: Threshold for "In Distribution"
        # Since we use Cosine Distance (0-2), 0.5 is a reasonable loose threshold for "same semantics"
        percentile_95 = 0.5

        # Extract constraints using the ALL-NEW ConstraintExtractor
        constraints = self.extractor.extract(
            retrieved_metadata=baseline_retrieved,
            img_emb=img_emb,
            txt_emb=txt_emb,
            centroid_distances=centroid_distances,
            query_distance=query_distance,
            percentile_95=percentile_95
        )

        return {
            "baseline": baseline,
            "no_text": no_text,
            "no_image": no_image,
            "noisy": noisy,
            "stability": stability,
            "constraints": constraints,
            "retrieved": baseline_retrieved,
        }
