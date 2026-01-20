"""
CounterfactualReasoner Orchestrator - UPDATED with constraints

Changes:
1. Pass constraints through to output
2. Maintain backward compatibility
3. Include constraints in explanation input
"""

class CounterfactualReasoner:
    def __init__(self, stability_runner, scorer, explainer=None):
        self.stability_runner = stability_runner
        self.scorer = scorer
        self.explainer = explainer  # Gemini (optional)

    def run(self, img_emb, txt_emb):
        """
        Run counterfactual reasoning with constraints.
        
        Returns stability, ranked hypotheses, constraints, and optional explanation.
        """
        # Run stability analysis (now includes constraints)
        stability = self.stability_runner.run(img_emb, txt_emb)
        
        # Score hypotheses
        ranked = self.scorer.score(stability)

        # Build output
        output = {
            "stability": stability,
            "ranked_hypotheses": [h.__dict__ for h in ranked],
        }
        
        # NEW: Include constraints if available
        if "constraints" in stability:
            output["constraints"] = stability["constraints"]
        
        # NEW: Include retrieved metadata if available
        if "retrieved" in stability:
            output["retrieved_metadata"] = stability["retrieved"]

        # Generate explanation if explainer provided
        if self.explainer:
            try:
                explanation = self.explainer.explain(output)
                output["explanation"] = explanation.__dict__
            except Exception as e:
                output["explanation_error"] = str(e)

        return output