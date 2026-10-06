"""Fail CI when a retrieval report falls below the thresholds in eval/thresholds.yaml."""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def latest_report(pattern: str) -> Path:
    reports = sorted(ROOT.glob(pattern))
    if not reports:
        sys.exit(f"No report matches {pattern} — run evaluate.py first.")
    return reports[-1]


def by_category(cases: list[dict], key: str) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for case in cases:
        groups.setdefault(case["category"], []).append(case["metrics"].get(key, 0.0))
    return {name: statistics.fmean(values) for name, values in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, help="Defaults to the newest matching report")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "eval" / "thresholds.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "vector_store" / "manifest.json")
    args = parser.parse_args()

    spec = yaml.safe_load(args.thresholds.read_text())
    report_path = args.report or latest_report(spec["report"])
    report = json.loads(report_path.read_text())
    summary, cases = report["summaries"][0], report["runs"][0]["cases"]

    # (check, expected, minimum, actual, passed)
    rows: list[tuple[str, str, str, str, bool]] = []

    def expect(name: str, wanted, actual) -> None:
        rows.append((name, str(wanted), "=", str(actual), wanted == actual))

    def floor(name: str, entry: dict, tolerance: float, actual: float | None) -> None:
        minimum = entry["measured"] - tolerance
        passed = actual is not None and actual >= minimum - 1e-9
        shown = "missing" if actual is None else f"{actual:.4f}"
        rows.append((name, f"{entry['measured']:.4f}", f"{minimum:.4f}", shown, passed))

    # The gate only means something for the configuration the baseline was measured on.
    expected = spec["expect"]
    expect("questions", expected["questions"], summary["questions"])
    for key, wanted in expected["config"].items():
        expect(f"config.{key}", wanted, summary["config"].get(key))
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else {}
    for key, wanted in expected["manifest"].items():
        expect(f"store.{key}", wanted, manifest.get(key))

    tolerance = spec["tolerance"]
    for key, entry in spec["metrics"].items():
        floor(key, entry, tolerance["aggregate"], summary.get(key))

    # Small categories move a lot per question, which is exactly why they get their
    # own floor: a regression confined to one category cannot hide in the mean.
    actual_categories = by_category(cases, "ndcg")
    for category in sorted(set(spec["category_ndcg"]) | set(actual_categories)):
        entry = spec["category_ndcg"].get(category)
        if entry is None:
            rows.append((f"ndcg[{category}]", "no baseline", "-", f"{actual_categories[category]:.4f}", False))
            continue
        floor(f"ndcg[{category}]", entry, tolerance["category"], actual_categories.get(category))

    failures = sum(not passed for *_, passed in rows)
    headers = ("check", "baseline", "min", "actual", "status")
    table = [(*row[:4], "PASS" if row[4] else "FAIL") for row in rows]

    widths = [max(len(cell) for cell in column) for column in zip(headers, *table)]
    print(f"Report: {report_path.relative_to(ROOT) if report_path.is_relative_to(ROOT) else report_path}\n")
    for line in [headers, tuple("-" * w for w in widths), *table]:
        print("  ".join(cell.ljust(w) for cell, w in zip(line, widths)).rstrip())
    verdict = f"{failures} of {len(rows)} checks failed" if failures else f"All {len(rows)} checks passed"
    print(f"\n{verdict}")

    if summary_file := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(summary_file, "a", encoding="utf-8") as handle:
            handle.write(f"## Retrieval quality gate — {'❌ failed' if failures else '✅ passed'}\n\n")
            handle.write(f"`{report_path.name}` · {verdict}\n\n")
            handle.write("| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n")
            for row in table:
                status = "✅ PASS" if row[4] == "PASS" else "❌ FAIL"
                handle.write("| " + " | ".join((*row[:4], status)) + " |\n")
            handle.write("\n")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
