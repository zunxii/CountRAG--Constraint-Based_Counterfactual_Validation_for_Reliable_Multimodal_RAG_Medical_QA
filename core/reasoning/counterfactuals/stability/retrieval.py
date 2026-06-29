from __future__ import annotations

import faiss
import numpy as np


class StabilityRetriever:
    def __init__(self, index, metadata):
        self.index = index
        self.metadata = metadata

    def retrieve(self, emb, top_k=10):
        q = emb.detach().cpu().numpy().astype("float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)

        results = []
        for s, i in zip(scores[0], idxs[0]):
            if i < 0 or i >= len(self.metadata):
                continue

            meta = self.metadata[int(i)]
            results.append({
                "diagnosis_label": meta.get("diagnosis_label", "unknown"),
                "score": float(s),
                "metadata": meta,
            })

        return results

    def posterior_distribution(self, emb, temperature: float = 0.05):
        q = emb.detach().cpu().numpy().astype("float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, self.index.ntotal)

        raw = scores[0].astype(np.float64)
        z = (raw - raw.max()) / max(temperature, 1e-6)
        z = z - z.max()
        probs = np.exp(z)
        probs = probs / (probs.sum() + 1e-12)

        dist = {}
        for p, i in zip(probs, idxs[0]):
            if i < 0 or i >= len(self.metadata):
                continue
            meta = self.metadata[int(i)]
            label = meta.get("diagnosis_label", "unknown")
            dist[label] = dist.get(label, 0.0) + float(p)

        return {
            "distribution": dist,
            "num_clusters": len(dist),
            "top_label": max(dist, key=dist.get) if dist else "unknown",
            "top_prob": max(dist.values()) if dist else 0.0,
        }