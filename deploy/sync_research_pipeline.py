"""
deploy/sync_research_pipeline.py
============================================================================
Re-copies the vendored research-pipeline modules from the original research
project into backend/research_pipeline/.

The backend imports those modules' own validated feature-construction
functions rather than reimplementing them, so the repository carries verbatim
copies (see backend/research_pipeline/README.md). They must never be edited
in place here -- if a research module genuinely changes, re-run this so the
whole set moves together:

    python deploy/sync_research_pipeline.py --source <path-to>/OpenAQ_API

--check reports drift without writing anything (useful in CI or before a
release).

ONE SANCTIONED TRANSFORMATION
-----------------------------
The research scripts hardcode the author's Google Cloud project id for their
own standalone `main()` entry points:

    EE_PROJECT_ID = "<the author's Google Cloud project>"

That identifies a billable project and this repository is public, so the copy
step rewrites exactly that assignment to read the environment instead. The
backend never calls those `main()` functions -- it imports the modules for
their feature-construction helpers -- so behaviour is unchanged either way.
The rewrite is applied on both copy and --check, which keeps the comparison
exact; nothing else about the files is touched.
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent.parent / "backend" / "research_pipeline"

# The five modules the backend imports, plus their transitive local imports.
MODULES = [
    "additional_features_loader", "advanced_feature_engineering_experiment",
    "build_additional_features", "combined_physics_features_loader", "diagnostics_core",
    "extract_era5_hourly_batched", "feature_ablation_experiment", "feature_groups",
    "optimize_and_validate", "physics_transport_ablation_experiment", "prepare_features",
    "run_logspace_pipeline_experiment", "run_logspace_r2_improvement_experiment",
    "run_physics_transport_experiment", "run_source_reweighting_experiment",
    "run_transfer_learning_experiment", "train_baseline_models", "validation_strategies",
]


# `EE_PROJECT_ID = "<anything>"` at the start of a line, with any trailing comment.
_PROJECT_ID_RE = re.compile(r'^EE_PROJECT_ID\s*=\s*["\'][^"\']*["\'].*$', re.MULTILINE)
_SCRUBBED = ('EE_PROJECT_ID = __import__("os").environ.get("PM25_GEE_PROJECT_ID", "")'
             '  # scrubbed on vendoring -- see deploy/sync_research_pipeline.py')


def scrub(text):
    """Replace a hardcoded Earth Engine project id with an environment read.
    `__import__("os")` keeps this a single-line, self-contained substitution:
    neither module imports os at the top, and adding an import would be a
    second, larger deviation from the source."""
    return _PROJECT_ID_RE.sub(_SCRUBBED, text)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def source_bytes(path):
    """The source file as it SHOULD appear once vendored."""
    return scrub(path.read_text(encoding="utf-8")).encode("utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path, help="the research project's OpenAQ_API directory")
    ap.add_argument("--check", action="store_true", help="report drift, write nothing")
    args = ap.parse_args()

    if not args.source.is_dir():
        sys.exit(f"--source {args.source} is not a directory")

    missing, drifted, copied = [], [], []
    for name in MODULES:
        src, dst = args.source / f"{name}.py", VENDOR_DIR / f"{name}.py"
        if not src.exists():
            missing.append(name)
            continue
        want = source_bytes(src)
        if dst.exists() and digest_bytes(want) == digest(dst):
            continue
        drifted.append(name)
        if not args.check:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(want)
            copied.append(name)

    if missing:
        sys.exit(f"ERROR: not found in {args.source}: {', '.join(missing)}")
    if args.check:
        if drifted:
            print(f"DRIFT: {len(drifted)} vendored module(s) differ from the source:")
            for n in drifted:
                print(f"  {n}.py")
            sys.exit(1)
        print(f"OK: all {len(MODULES)} vendored modules match {args.source}")
        return
    print(f"Updated {len(copied)} of {len(MODULES)} module(s)."
          if copied else f"OK: all {len(MODULES)} modules already current.")


if __name__ == "__main__":
    main()
