"""Run every samples/*.txt through the extractor and print a summary.

Usage: python -m scripts.run_samples
"""
import sys
import time
from pathlib import Path

from app.extractor import ExtractorError, extract_job_offer

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def main() -> int:
    files = sorted(SAMPLES.glob("*.txt"))
    if not files:
        print(f"No .txt files in {SAMPLES}")
        return 1

    failures = 0
    for f in files:
        print(f"\n=== {f.name} ({len(f.read_text())} chars) ===")
        start = time.time()
        try:
            offer = extract_job_offer(f.read_text())
            print(offer.model_dump_json(indent=2, exclude_none=True))
        except ExtractorError as e:
            failures += 1
            print(f"FAILED ({type(e).__name__}): {e}")
            for d in getattr(e, "details", []):
                print(f"  - {d['field']}: {d['error']}")
        print(f"[{time.time() - start:.1f}s]")

    print(f"\n{len(files) - failures}/{len(files)} extracted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
