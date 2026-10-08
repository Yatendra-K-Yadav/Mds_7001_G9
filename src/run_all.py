"""
Run the whole analysis in order.

    python -m src.run_all              # everything
    python -m src.run_all --figures    # skip straight to the charts

Individual steps can also be run on their own, e.g.

    python -m src.analysis.regression
"""

import argparse
import importlib
import sys
import time

# (module, what it does) in dependency order.
STEPS = [
    ("src.pipeline.ingest", "Build analysis_table.csv from the five raw sources"),
    ("src.pipeline.quality", "Coverage and suppression report"),
    ("src.analysis.bias", "Who is missing from the data, and is it systematic?"),
    ("src.analysis.regression", "Cross-sectional OLS, year by year"),
    ("src.analysis.forest", "Random Forest vs OLS"),
    ("src.analysis.pooled", "Pool years per council"),
    ("src.analysis.clustering", "k-means council types"),
    ("src.figures.headline", "Figure 1 - density vs disadvantage"),
    ("src.figures.trend", "Figure 2 - the gap over time"),
    ("src.figures.clusters", "Figure 3 - council types"),
]

FIGURE_STEPS = [s for s in STEPS if s[0].startswith("src.figures")]


def run(steps, quiet):
    failures = []
    for index, (module_name, description) in enumerate(steps, start=1):
        print(f"\n{'=' * 74}")
        print(f"[{index}/{len(steps)}] {module_name}")
        print(f"        {description}")
        print("=" * 74)
        started = time.time()
        try:
            module = importlib.import_module(module_name)
            if quiet:
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main()
            else:
                module.main()
            print(f"  done in {time.time() - started:.1f}s")
        except Exception as error:            # noqa: BLE001 - report and continue
            failures.append((module_name, error))
            print(f"  FAILED: {type(error).__name__}: {error}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true",
                        help="only regenerate the figures")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress each step's own output")
    args = parser.parse_args()

    steps = FIGURE_STEPS if args.figures else STEPS
    failures = run(steps, args.quiet)

    print(f"\n{'=' * 74}")
    if failures:
        print(f"{len(failures)} step(s) failed:")
        for module_name, error in failures:
            print(f"  - {module_name}: {error}")
        sys.exit(1)
    print(f"All {len(steps)} steps completed.")


if __name__ == "__main__":
    main()
