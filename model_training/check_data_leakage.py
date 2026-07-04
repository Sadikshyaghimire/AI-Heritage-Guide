import os
import hashlib

BASE_DIR = "dataset/final_dataset"

SETS = ["train", "val", "test"]


def get_md5(file_path):
    """Return MD5 hash of a file."""
    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def build_dataset_index(dataset_name):
    """
    Builds dictionaries for filenames and hashes.

    Returns:
        filenames[class] -> set(filename)
        hashes[class][md5] -> {
            "filename": ...,
            "path": ...
        }
    """

    dataset_path = os.path.join(BASE_DIR, dataset_name)

    filenames = {}
    hashes = {}

    for class_name in sorted(os.listdir(dataset_path)):

        class_path = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_path):
            continue

        filenames[class_name] = set()
        hashes[class_name] = {}

        for file in os.listdir(class_path):

            filepath = os.path.join(class_path, file)

            if not os.path.isfile(filepath):
                continue

            filenames[class_name].add(file)

            md5 = get_md5(filepath)

            hashes[class_name][md5] = {
                "filename": file,
                "path": filepath
            }

    return filenames, hashes


# ----------------------------------------------------
# Build indices
# ----------------------------------------------------

indices = {}

for dataset in SETS:
    indices[dataset] = build_dataset_index(dataset)

print("=" * 70)
print("DATA LEAKAGE VERIFICATION")
print("=" * 70)

pairs = [
    ("train", "val"),
    ("train", "test"),
    ("val", "test")
]

total_duplicates = 0

for first, second in pairs:

    print(f"\nChecking {first.upper()} <--> {second.upper()}")

    first_names, first_hashes = indices[first]
    second_names, second_hashes = indices[second]

    pair_duplicates = 0

    for cls in sorted(first_names.keys()):

        # -------------------------
        # Filename duplicates
        # -------------------------

        duplicate_names = (
            first_names[cls]
            .intersection(second_names[cls])
        )

        if duplicate_names:

            print(f"\nFilename duplicates in class '{cls}'")

            for name in sorted(duplicate_names):
                print("   ", name)

            pair_duplicates += len(duplicate_names)

        # -------------------------
        # Image duplicates (MD5)
        # -------------------------

        duplicate_hashes = (
            set(first_hashes[cls].keys())
            .intersection(second_hashes[cls].keys())
        )

        if duplicate_hashes:

            print(f"\nImage duplicates in class '{cls}'")

            for md5 in duplicate_hashes:

                print("-" * 60)
                print("MD5:", md5)

                print("\nFirst dataset:")
                print(first_hashes[cls][md5]["path"])

                print("\nSecond dataset:")
                print(second_hashes[cls][md5]["path"])

                print("-" * 60)

            pair_duplicates += len(duplicate_hashes)

    if pair_duplicates == 0:
        print("✅ No leakage detected.")
    else:
        print(f"❌ {pair_duplicates} duplicated images found!")

    total_duplicates += pair_duplicates

print("\n" + "=" * 70)

if total_duplicates == 0:
    print("🎉 DATASET PASSED.")
    print("No train / validation / test leakage detected.")
else:
    print("⚠ DATASET FAILED.")
    print(f"{total_duplicates} duplicated images found.")

print("=" * 70)