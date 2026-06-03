import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "raw" / "solodit" / "solodit.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "processed" / "solodit_vulnerable_filtered.json"


def extract_vulnerable_samples(input_path: Path, output_path: Path) -> int:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = []
    for sample in data:
        if str(sample.get("label", "")).strip().lower() != "vulnerable":
            continue

        filtered.append(
            {
                "Swc": sample.get("swc"),
                "Title": sample.get("title"),
                "Description": sample.get("description"),
                "Remediation": sample.get("remediation"),
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    return len(filtered)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter vulnerable samples from Solodit JSON and keep only "
            "Swc, Title, Description, Remediation fields."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to input solodit JSON file",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to output JSON file",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    count = extract_vulnerable_samples(input_path, output_path)
    print(f"Created: {output_path} ({count} records)")


if __name__ == "__main__":
    main()
