import json
import google.generativeai as genai
from .prompt import PROMPT
from .schemas import Explanation


class GeminiCounterfactualExplainer:
    """
    Layer 3: Explanation only.
    NEVER touches embeddings or retrieval.
    Pure interpretation of structured signals.
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Gemini API key not provided")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def explain(self, structured_input: dict) -> Explanation:
        prompt = PROMPT.format(
            input_json=json.dumps(structured_input, indent=2)
        )

        response = self.model.generate_content(
            prompt,
        )

        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()

        try:
            data = json.loads(text)
        except Exception as e:
            raise RuntimeError(
                f"Gemini returned invalid JSON:\n{text}"
            ) from e

        return Explanation(**data)
