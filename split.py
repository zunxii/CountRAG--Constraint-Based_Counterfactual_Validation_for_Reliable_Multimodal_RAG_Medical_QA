import csv
import random
from pathlib import Path
from collections import defaultdict

# ============================
# CONFIG
# ============================
INPUT_CSV = "data/raw/clipsyntel.csv"
OUTPUT_DIR = "data/processed/splits"
TRAIN_RATIO = 0.8
SEED = 42

IMAGE_COL = "image_path"
CATEGORY_COL = "category"

random.seed(SEED)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

TRAIN_CSV = Path(OUTPUT_DIR) / "train.csv"
EVAL_CSV  = Path(OUTPUT_DIR) / "eval.csv"

# ============================
# LOAD CSV
# ============================
rows = []
with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        rows.append(row)

total_rows = len(rows)
target_train_rows = int(total_rows * TRAIN_RATIO)

print(f"✓ Loaded {total_rows} total rows")
print(f"✓ Target train rows: ~{target_train_rows}")

# ============================
# GROUP: category → image → rows
# ============================
cat_img_rows = defaultdict(lambda: defaultdict(list))
for r in rows:
    cat_img_rows[r[CATEGORY_COL]][r[IMAGE_COL]].append(r)

# ============================
# PHASE 1: CATEGORY SEEDING
# ============================
train_rows, eval_rows = [], []
train_images, eval_images = set(), set()

remaining_images = set()

for category, img_map in cat_img_rows.items():
    images = list(img_map.keys())
    if len(images) < 2:
        raise ValueError(
            f"Category '{category}' has <2 images — cannot split safely"
        )

    random.shuffle(images)

    eval_img = images[0]
    train_img = images[1]

    eval_rows.extend(img_map[eval_img])
    train_rows.extend(img_map[train_img])

    eval_images.add(eval_img)
    train_images.add(train_img)

    for img in images[2:]:
        remaining_images.add(img)

print("✓ Phase 1 complete (all categories seeded)")

# ============================
# PHASE 2: FILL TO ~80% ROWS
# ============================
remaining_images = list(remaining_images)
random.shuffle(remaining_images)

for img in remaining_images:
    if len(train_rows) < target_train_rows:
        train_rows.extend(
            next(iter(
                cat_img_rows[cat][img]
                for cat in cat_img_rows
                if img in cat_img_rows[cat]
            ))
        )
        train_images.add(img)
    else:
        eval_rows.extend(
            next(iter(
                cat_img_rows[cat][img]
                for cat in cat_img_rows
                if img in cat_img_rows[cat]
            ))
        )
        eval_images.add(img)

# ============================
# SAFETY CHECKS
# ============================
assert train_images.isdisjoint(eval_images), "❌ Image leakage detected"

train_cats = {r[CATEGORY_COL] for r in train_rows}
eval_cats  = {r[CATEGORY_COL] for r in eval_rows}

missing = train_cats.symmetric_difference(eval_cats)
if missing:
    raise RuntimeError(f"❌ Category missing in one split: {missing}")

# ============================
# WRITE CSVs
# ============================
def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

write_csv(TRAIN_CSV, train_rows)
write_csv(EVAL_CSV, eval_rows)

# ============================
# SUMMARY
# ============================
print("\n✅ ROW-WISE SPLIT COMPLETE")
print(f"Train rows: {len(train_rows)} ({len(train_rows)/total_rows:.2%})")
print(f"Eval  rows: {len(eval_rows)} ({len(eval_rows)/total_rows:.2%})")
print(f"Train CSV: {TRAIN_CSV}")
print(f"Eval  CSV: {EVAL_CSV}")
print("✓ No image overlap")
print("✓ All categories present in both splits")
