"""
Generate labeled training pairs from the PAN-2011 Plagiarism Corpus.

This script reads the PAN external-detection corpus:
  data/pan-plagiarism-corpus-2011/external-detection-corpus/
    suspicious-document/partN/suspicious-documentXXXXX.txt  (+ .xml)
    source-document/partN/source-documentXXXXX.txt

The XML annotation tells us which (suspicious_doc, source_doc) pairs
contain confirmed plagiarism passages.  We create:
  - POSITIVE pairs: suspicious doc paired with its actual source  (label=1)
  - NEGATIVE pairs: suspicious doc paired with a random non-source (label=0)

Output:
  data/pairs.csv  — columns: doc_a, doc_b, label
  data/documents/ — copies of all text files used
"""
import sys
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CORPUS_ROOT = Path("data/pan-plagiarism-corpus-2011/external-detection-corpus")
SUSPICIOUS_ROOT = CORPUS_ROOT / "suspicious-document"
SOURCE_ROOT = CORPUS_ROOT / "source-document"
OUTPUT_DOCS_DIR = Path("data/documents")
OUTPUT_PAIRS_CSV = Path("data/pairs.csv")

# How many suspicious documents to sample (to keep training time reasonable)
MAX_SUSPICIOUS_DOCS = 200
NEGATIVES_PER_POSITIVE = 2
RANDOM_SEED = 42


def find_source_references(xml_path: Path) -> list[str]:
    """Parse XML and return source document references for this suspicious doc."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        sources = set()
        for feat in root.findall("feature"):
            if feat.get("name") == "plagiarism":
                src = feat.get("source_reference")
                if src:
                    sources.add(src)
        return list(sources)
    except Exception:
        return []


def find_file_in_parts(root_dir: Path, filename: str) -> Path | None:
    """Search for a file across all part subdirectories."""
    for part_dir in sorted(root_dir.iterdir()):
        if not part_dir.is_dir():
            continue
        candidate = part_dir / filename
        if candidate.exists():
            return candidate
    return None


def collect_all_source_files() -> list[Path]:
    """Return all .txt source files in the corpus."""
    sources = []
    for part_dir in sorted(SOURCE_ROOT.iterdir()):
        if part_dir.is_dir():
            sources.extend(sorted(part_dir.glob("*.txt")))
    return sources


def main():
    random.seed(RANDOM_SEED)

    OUTPUT_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Gather suspicious documents
    suspicious_files = []
    for part_dir in sorted(SUSPICIOUS_ROOT.iterdir()):
        if part_dir.is_dir():
            suspicious_files.extend(sorted(part_dir.glob("*.txt")))

    if not suspicious_files:
        print("ERROR: No suspicious documents found. Check corpus path.")
        return

    print(f"Found {len(suspicious_files)} suspicious documents.")

    # Sample if too many
    if len(suspicious_files) > MAX_SUSPICIOUS_DOCS:
        suspicious_files = random.sample(suspicious_files, MAX_SUSPICIOUS_DOCS)
        print(f"Sampled {MAX_SUSPICIOUS_DOCS} for training.")

    # Gather all source files for negative sampling
    all_source_files = collect_all_source_files()
    if not all_source_files:
        print("ERROR: No source documents found. Check corpus path.")
        return
    print(f"Found {len(all_source_files)} source documents.")

    source_name_to_path = {f.name: f for f in all_source_files}

    # Build pairs
    pairs = []
    copied_docs = set()

    def copy_doc(src_path: Path) -> str:
        """Copy doc to output dir and return filename."""
        dest = OUTPUT_DOCS_DIR / src_path.name
        if src_path.name not in copied_docs:
            shutil.copy2(src_path, dest)
            copied_docs.add(src_path.name)
        return src_path.name

    positive_count = 0
    negative_count = 0

    for susp_txt in suspicious_files:
        xml_path = susp_txt.with_suffix(".xml")
        if not xml_path.exists():
            continue

        source_refs = find_source_references(xml_path)
        if not source_refs:
            continue

        # Resolve source files
        source_paths = []
        for ref in source_refs:
            found = find_file_in_parts(SOURCE_ROOT, ref) or source_name_to_path.get(ref)
            if found:
                source_paths.append(found)

        if not source_paths:
            continue

        susp_name = copy_doc(susp_txt)

        # Positive pairs
        for src_path in source_paths[:1]:  # Use 1 source per suspicious doc
            src_name = copy_doc(src_path)
            pairs.append((susp_name, src_name, 1))
            positive_count += 1

            # Negative pairs: random sources that are NOT actual sources
            actual_source_names = {Path(ref).name for ref in source_refs}
            candidate_negatives = [
                f for f in all_source_files
                if f.name not in actual_source_names
            ]
            if candidate_negatives:
                neg_sources = random.sample(
                    candidate_negatives,
                    min(NEGATIVES_PER_POSITIVE, len(candidate_negatives))
                )
                for neg_src in neg_sources:
                    neg_name = copy_doc(neg_src)
                    pairs.append((susp_name, neg_name, 0))
                    negative_count += 1

    if not pairs:
        print("No pairs generated. Check corpus structure.")
        return

    # Write CSV
    import csv
    with open(OUTPUT_PAIRS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_a", "doc_b", "label"])
        for doc_a, doc_b, label in pairs:
            writer.writerow([doc_a, doc_b, label])

    print(f"\nGenerated {len(pairs)} pairs:")
    print(f"  Positive (plagiarism): {positive_count}")
    print(f"  Negative (non-plagiarism): {negative_count}")
    print(f"  Documents copied to: {OUTPUT_DOCS_DIR}/")
    print(f"  Pairs CSV written to: {OUTPUT_PAIRS_CSV}")
    print(f"\nNext step: python training/train_model.py")


if __name__ == "__main__":
    main()
