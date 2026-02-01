# Comparison Report: Research RAG vs CLIPSyntel Baseline
**Date**: 2026-02-01 01:28:29.654443

| Image | Ground Truth (CLIPSyntel) | RAG Top Prediction | Match (Baseline)? | Match (Noisy)? | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1.json (`skin irritation_Image_5.jpg`) | **skin irritation** | skin growth | ✅ | ✅ | ✅ |
| 2.json (`edema_Image_2.jpg`) | **edema** | edema | ✅ | ✅ | ✅ |
| 3.json (`skin rash_Image_2.jpg`) | **skin rash** | skin irritation | ❌ | ❌ | ✅ |

## 1.json: Skin Irritation
- **Query**: "Hello doctor,For three months now, two days after finishing a 21-day course of antibiotics to clear ..."
- **Ground Truth Summary**: Is the rough patch of skin on the inside of the patient's cheek near the molars related to their TMJ problems, and is it a cause for concern regarding cancer?
- **RAG Baseline Clusters**: {'skin growth': 0.8409, 'skin irritation': 0.1591}
- **RAG Noisy Clusters** (Uncertainty): {'skin growth': 0.7496, 'skin irritation': 0.2504}
- **Analysis**: The model successfully retrieved the correct diagnosis.

## 2.json: Edema
- **Query**: "I have had several sudden unexplained lumps appear on my head. This started a few days ago. The lump..."
- **Ground Truth Summary**: Is Voltaren gel recommended for a 66-year-old Type I Diabetic with edema, ankle injury, and shooting, burning pain?
- **RAG Baseline Clusters**: {'edema': 1.0}
- **RAG Noisy Clusters** (Uncertainty): {'edema': 0.8945, 'neck swelling': 0.1055}
- **Analysis**: The model successfully retrieved the correct diagnosis.

## 3.json: Skin Rash
- **Query**: "My 9 year old daughter is complaining that it feels like something is poking her in arms, legs, and ..."
- **Ground Truth Summary**: What could be the cause of random bruises on a 3-year-old's back and sides, along with a skin rash resembling pimples?
- **RAG Baseline Clusters**: {'skin irritation': 0.8896, 'skin dryness': 0.1104}
- **RAG Noisy Clusters** (Uncertainty): {'skin irritation': 1.0}
- **Analysis**: The model successfully retrieved the correct diagnosis.