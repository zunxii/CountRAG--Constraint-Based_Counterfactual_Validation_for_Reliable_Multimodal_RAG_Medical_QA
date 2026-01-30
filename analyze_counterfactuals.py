
import json
from collections import defaultdict

def analyze(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    samples = data['stability_tests']
    
    # Metrics to track
    jsd_no_text = []
    jsd_no_image = []
    jsd_noisy = []
    
    robustness_counts = defaultdict(int)
    
    # Diagnosis specific
    diag_jsd = defaultdict(list)
    
    for s in samples:
        stab = s['stability']
        jsd = stab['js_divergence']
        
        jsd_no_text.append(jsd['no_text'])
        jsd_no_image.append(jsd['no_image'])
        jsd_noisy.append(jsd['noisy'])
        
        robustness_counts[stab['robustness_level']] += 1
        
        diag_jsd[s['diagnosis']].append(jsd)

    # Averages
    avg_no_text = sum(jsd_no_text) / len(jsd_no_text)
    avg_no_image = sum(jsd_no_image) / len(jsd_no_image)
    avg_noisy = sum(jsd_noisy) / len(jsd_noisy)
    
    print(f"Total Samples: {len(samples)}")
    print("-" * 30)
    print(f"Average JSD (No Text): {avg_no_text:.4f}")
    print(f"Average JSD (No Image): {avg_no_image:.4f}")
    print(f"Average JSD (Noisy): {avg_noisy:.4f}")
    print("-" * 30)
    print("Robustness Levels:")
    for level, count in robustness_counts.items():
        print(f"{level}: {count} ({count/len(samples)*100:.2f}%)")
        
    print("-" * 30)
    print("Diagnosis Specific (Avg JSD No Text / No Image):")
    target_diagnoses = ["swollen tonsils", "neck swelling", "lip swelling", "mouth ulcers", "knee swelling", "skin rash", "edema"]
    # fuzzy match or exact? The json has lowercase.
    
    for d in target_diagnoses:
        if d in diag_jsd:
            vals = diag_jsd[d]
            nt = sum(x['no_text'] for x in vals) / len(vals)
            ni = sum(x['no_image'] for x in vals) / len(vals)
            print(f"{d}: NoText={nt:.4f}, NoImage={ni:.4f}")

if __name__ == "__main__":
    analyze("/Users/faqrealam149/Desktop/research-rag/outputs/evaluation/counterfactual/results.json")
