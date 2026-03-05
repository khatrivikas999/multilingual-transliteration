import os
import json
import zipfile
import glob
import pandas as pd
from sklearn.model_selection import train_test_split


HF_CACHE = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "huggingface",
    "hub"
)


LANG_ZIPS = {
    "hi": "hin.zip",
    "bn": "ben.zip",
    "ta": "tam.zip"
}


def find_dataset_dir():

    base = os.path.join(HF_CACHE, "datasets--ai4bharat--Aksharantar")

    if not os.path.exists(base):
        raise RuntimeError("Dataset not found in HF cache")

    snapshots = glob.glob(os.path.join(base, "snapshots", "*"))

    if not snapshots:
        raise RuntimeError("No snapshot directory found")

    return snapshots[0]


def load_language(zip_path):

    rows = []

    with zipfile.ZipFile(zip_path, "r") as z:

        for name in z.namelist():

            if not name.endswith(".json"):
                continue

            with z.open(name) as f:

                for line in f:
                    obj = json.loads(line.decode("utf-8"))

                    # Remove problematic column
                    obj.pop("score", None)

                    rows.append(obj)

    return rows


def prepare_data(output_dir="data"):

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Loading Aksharantar Dataset (Manual Loader)")
    print("=" * 70)

    # ------------------------------------------------
    # Locate HF snapshot
    # ------------------------------------------------

    print("\nFinding dataset files...")

    dataset_dir = find_dataset_dir()

    print("✓ Found dataset at:", dataset_dir)

    # ------------------------------------------------
    # Load selected languages
    # ------------------------------------------------

    all_rows = []

    for code, zip_name in LANG_ZIPS.items():

        path = os.path.join(dataset_dir, zip_name)

        if not os.path.exists(path):
            raise RuntimeError(f"{zip_name} not found")

        print(f"Loading {zip_name}...")

        rows = load_language(path)

        print(f"✓ {len(rows)} samples")

        for r in rows:
            r["lang"] = code

        all_rows.extend(rows)

    print(f"\nTotal samples: {len(all_rows)}")

    # ------------------------------------------------
    # Create DataFrame
    # ------------------------------------------------

    df = pd.DataFrame(all_rows)

    print("Columns:", df.columns.tolist())

    SRC = "english word"
    TGT = "native word"

    if SRC not in df or TGT not in df:
        raise RuntimeError("Required columns missing")

    # ------------------------------------------------
    # Train / Test split per language
    # ------------------------------------------------

    all_train = []
    all_test = []

    for code in LANG_ZIPS.keys():

        print(f"\nProcessing {code}")

        lang_df = df[df["lang"] == code]

        train_df, test_df = train_test_split(
            lang_df,
            test_size=0.15,
            random_state=42,
            shuffle=True
        )

        for _, row in train_df.iterrows():

            src = str(row[SRC]).strip()
            tgt = str(row[TGT]).strip()

            if src and tgt:

                all_train.append({
                    "source": f"<{code}> {src}",
                    "target": tgt
                })

        for _, row in test_df.iterrows():

            src = str(row[SRC]).strip()
            tgt = str(row[TGT]).strip()

            if src and tgt:

                all_test.append({
                    "source": f"<{code}> {src}",
                    "target": tgt
                })

        print(f"✓ {len(train_df)} train / {len(test_df)} test")

    # ------------------------------------------------
    # Save
    # ------------------------------------------------

    if not all_train:
        raise RuntimeError("No data generated")

    print("\nSaving output files...")

    train_file = os.path.join(output_dir, "train.jsonl")
    test_file = os.path.join(output_dir, "test.jsonl")

    with open(train_file, "w", encoding="utf-8") as f:
        for x in all_train:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(test_file, "w", encoding="utf-8") as f:
        for x in all_test:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # ------------------------------------------------
    # Stats
    # ------------------------------------------------

    print("\n✓ Done")
    print("Train:", len(all_train))
    print("Test :", len(all_test))

    print("\nSample:")

    for i in range(min(5, len(all_train))):
        print(all_train[i]["source"])
        print("→", all_train[i]["target"])

    print("\nReady for training 🚀")

    return True


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    prepare_data()
