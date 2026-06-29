"""
Retrieval metrics computation
"""
import numpy as np
from typing import List, Dict


class MetricsCalculator:
    def compute_all_metrics(self, retrieved: List[Dict], target_label: str) -> Dict:
        hits = [
            1 if r.get("diagnosis_label") == target_label else 0
            for r in retrieved
        ]
        scores = np.array([float(r.get("score", 0.0)) for r in retrieved], dtype=np.float32)

        metrics = {}
        metrics.update(self._recall_precision(hits))
        metrics["MRR"] = self._mrr(hits)
        metrics["MAP"] = self._map(hits)
        metrics.update(self._ndcg(hits))
        metrics["Entropy"] = self._entropy(scores)
        metrics["Margin"] = self._margin(scores)
        return metrics

    def _recall_precision(self, hits: List[int]) -> Dict:
        metrics = {}
        for k in [1, 5, 10, 20]:
            if k <= len(hits):
                topk = hits[:k]
                metrics[f"R@{k}"] = 1.0 if sum(topk) > 0 else 0.0
                metrics[f"P@{k}"] = sum(topk) / k
        return metrics

    def _mrr(self, hits: List[int]) -> float:
        for i, h in enumerate(hits):
            if h == 1:
                return 1.0 / (i + 1)
        return 0.0

    def _map(self, hits: List[int]) -> float:
        relevant_count = 0
        precision_sum = 0.0
        for i, h in enumerate(hits):
            if h == 1:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)

        total_relevant = sum(hits)
        return precision_sum / total_relevant if total_relevant > 0 else 0.0

    def _ndcg(self, hits: List[int]) -> Dict:
        metrics = {}
        for k in [5, 10]:
            if k <= len(hits):
                dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits[:k]))
                ideal = sorted(hits[:k], reverse=True)
                idcg = sum(h / np.log2(i + 2) for i, h in enumerate(ideal))
                metrics[f"NDCG@{k}"] = dcg / idcg if idcg > 0 else 0.0
        return metrics

    def _entropy(self, scores: np.ndarray, temperature: float = 0.05) -> float:
        """
        Score entropy over a sharpened softmax of retrieval scores.
        Temperature < 1 makes score differences matter more.
        """
        if scores.size == 0:
            return 0.0

        z = scores - scores.max()
        z = z / max(temperature, 1e-6)
        z = z - z.max()

        probs = np.exp(z)
        probs = probs / (probs.sum() + 1e-12)
        return float(-(probs * np.log2(probs + 1e-12)).sum())

    def _margin(self, scores: np.ndarray) -> float:
        """
        Raw score gap between the top-1 and top-2 retrieved items.
        """
        if scores.size < 2:
            return 0.0

        sorted_scores = np.sort(scores)[::-1]
        return float(sorted_scores[0] - sorted_scores[1])