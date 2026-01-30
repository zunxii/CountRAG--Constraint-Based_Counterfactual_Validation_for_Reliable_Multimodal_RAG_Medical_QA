import google.generativeai as genai
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retrieval.retriever import KBRetriever
from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.fusion.adaptive_fusion import AdaptiveFusion
from configs.inference_config import INFERENCE_CONFIG
import torch

def main():
    parser = argparse.ArgumentParser(description="Interpret a query using Gemini.")
    parser.add_argument("--query-text", required=True, help="Query text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    args = parser.parse_args()

    # Configure the Gemini API key
    try:
        # Assumes the API key is set in an environment variable
        genai.configure()
    except Exception as e:
        print(f"Please set the GOOGLE_API_KEY environment variable. {e}")
        return
        
    # Load the models and retriever
    print("Loading models and retriever...")
    encoder = BioMedCLIPEncoder(
        device=INFERENCE_CONFIG["device"],
        lora_path=INFERENCE_CONFIG.get("lora_path")
    )
    fusion = AdaptiveFusion().to(INFERENCE_CONFIG["device"])
    fusion.load_state_dict(
        torch.load(
            INFERENCE_CONFIG["fusion_path"],
            map_location=INFERENCE_CONFIG["device"]
        )
    )
    fusion.eval()
    retriever = KBRetriever(INFERENCE_CONFIG["kb_dir"])
    print("Models and retriever loaded.")

    # Encode the query
    print("Encoding query...")
    with torch.no_grad():
        txt_emb = encoder.encode_text(args.query_text).unsqueeze(0)
        img_emb = torch.zeros_like(txt_emb)
        query_emb = fusion(img_emb, txt_emb)
    print("Query encoded.")

    # Retrieve the top-k results
    print(f"Retrieving top-{args.top_k} results...")
    query_np = query_emb.cpu().numpy().astype("float32")
    scores, indices = retriever.search(query_np, args.top_k)
    print("Results retrieved.")

    # Get the clinical text from the results
    print("Extracting clinical text...")
    clinical_texts = []
    for idx in indices[0]:
        case = retriever.get_metadata(int(idx))
        if case:
            clinical_texts.append(case['clinical_text']['combined'])
    print("Clinical text extracted.")

    # Create the prompt for Gemini
    print("Creating prompt for Gemini...")
    prompt = "I have a patient with a swollen eye. Here are some similar cases from a medical database:\n\n"
    for i, text in enumerate(clinical_texts):
        prompt += f"Case {i+1}:\n{text}\n\n"
    prompt += "Based on these cases, what could be the possible causes of the swollen eye? Please provide a brief interpretation."
    print("Prompt created.")

    # Send the prompt to Gemini
    print("Sending prompt to Gemini...")
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    print("Response received.")

    # Print the interpretation
    print("\n\n--- Gemini Interpretation ---")
    print(response.text)
    print("---------------------------\n")

if __name__ == "__main__":
    main()
