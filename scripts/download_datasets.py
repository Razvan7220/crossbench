#!/usr/bin/env python3
"""Download and cache benchmark datasets locally."""

import argparse
from pathlib import Path


def download_humaneval():
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    print(f"HumanEval: {len(ds)} tasks cached")


def download_mbpp():
    from datasets import load_dataset
    ds = load_dataset("mbpp", "sanitized", split="test", trust_remote_code=True)
    print(f"MBPP: {len(ds)} tasks cached")


def download_bigcodebench():
    from datasets import load_dataset
    ds = load_dataset(
        "bigcode/bigcodebench",
        split="v0.1.2",
        revision="b74c0d0bf70d2c0bc459be537895cca163007f1a",
    )
    print(f"BigCodeBench: {len(ds)} tasks cached")


def download_livecodebench():
    from datasets import load_dataset
    ds = load_dataset("livecodebench/livecodebench", split="test", trust_remote_code=True)
    print(f"LiveCodeBench: {len(ds)} tasks cached")


def download_swebench():
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test", trust_remote_code=True)
    print(f"SWE-bench Verified: {len(ds)} instances cached")


def download_crosscodeeval():
    from datasets import load_dataset
    total = 0
    for lang in ("python", "java", "csharp", "typescript"):
        ds = load_dataset(f"ZHENGRAN/cross_code_eval_{lang}", split="train")
        total += len(ds)
        print(f"  {lang}: {len(ds)} examples")
    print(f"CrossCodeEval: {total} examples cached across 4 languages")


def download_repobench():
    from datasets import load_dataset
    total = 0
    for lang in ("python", "java"):
        ds_name = f"tianyang/repobench_{lang}_v1.1"
        for split in ("cross_file_first", "cross_file_random", "in_file"):
            ds = load_dataset(ds_name, split=split)
            total += len(ds)
            print(f"  {lang}/{split}: {len(ds)} examples")
    print(f"RepoBench v1.1: {total} examples cached across python+java")


def download_terminalbench():
    import subprocess
    data_dir = Path("data/terminal-bench")
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print("Terminal-Bench requires manual setup. Clone the repo:")
        print("  git clone https://github.com/terminal-bench/terminal-bench.git data/terminal-bench")
    else:
        print(f"Terminal-Bench data directory exists at {data_dir}")


DOWNLOADERS = {
    "humaneval": download_humaneval,
    "mbpp": download_mbpp,
    "bigcodebench": download_bigcodebench,
    "livecodebench": download_livecodebench,
    "swebench": download_swebench,
    "crosscodeeval": download_crosscodeeval,
    "repobench": download_repobench,
    "terminalbench": download_terminalbench,
}


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--benchmark", "-b",
        choices=["all"] + list(DOWNLOADERS.keys()),
        default="all",
        help="Which benchmark to download",
    )
    args = parser.parse_args()

    targets = DOWNLOADERS if args.benchmark == "all" else {args.benchmark: DOWNLOADERS[args.benchmark]}
    for name, fn in targets.items():
        print(f"\n--- {name} ---")
        try:
            fn()
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
