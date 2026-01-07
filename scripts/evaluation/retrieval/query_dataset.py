import pandas as pd

class ExternalQueryDataset:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)

        self.queries = []
        for _, row in df.iterrows():
            self.queries.append({
                "text": row["Question"],  # or Question_summ
                "image_path": row["image_path"],
                "label": row["category"]
            })

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx]
