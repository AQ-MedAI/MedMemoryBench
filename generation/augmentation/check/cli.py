#!/usr/bin/env python3
"""Query difficulty check CLI."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from generation.augmentation.check.config import CheckRunnerConfig, CheckerConfig, EnhancerConfig
from generation.augmentation.check.runner import CheckRunner


def main():
    parser = argparse.ArgumentParser(
        description="Query difficulty check - test if a memory-free model can answer questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m augmentation.check.cli
    python -m augmentation.check.cli --persona 1
    python -m augmentation.check.cli --persona 1 --persona 2 --persona 3
    python -m augmentation.check.cli --regenerate
    python -m augmentation.check.cli --report-format json
    python -m augmentation.check.cli --dry-run
        """,
    )

    _gen_dir = str(Path(__file__).parent.parent.parent.resolve())
    parser.add_argument("--dataset-dir", type=str,
        default=str(Path(_gen_dir) / "dataset"),
        help="Dataset directory path")
    parser.add_argument("--output-dir", type=str,
        default=str(Path(_gen_dir) / "augmentation"),
        help="Output directory path")
    parser.add_argument("--persona", type=int, action="append", dest="persona_ids",
        help="Persona ID(s) to check (repeatable)")
    parser.add_argument("--model", type=str, default=None,
        help="LLM model (defaults to .env config)")
    parser.add_argument("--temperature", type=float, default=1.0,
        help="LLM temperature (default: 1.0)")
    parser.add_argument("--regenerate", action="store_true",
        help="Enable regeneration for correctly-answered questions")
    parser.add_argument("--dry-run", action="store_true",
        help="Dry-run mode: check only, no file modifications")
    parser.add_argument("--no-report", action="store_true",
        help="Skip report generation")
    parser.add_argument("--report-format", type=str, choices=["markdown", "json", "both"],
        default="both", help="Report format (default: both)")
    parser.add_argument("--quiet", action="store_true",
        help="Reduce log output")
    parser.add_argument("--verbose", action="store_true", default=True,
        help="Show detailed logs")

    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    checker_config = CheckerConfig(
        model=args.model,
        temperature=args.temperature,
        verbose=not args.quiet,
    )

    enhancer_config = EnhancerConfig(
        model=args.model,
    )

    config = CheckRunnerConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        persona_ids=args.persona_ids or [],
        checker=checker_config,
        enhancer=enhancer_config,
        enable_regenerate=args.regenerate,
        dry_run=args.dry_run,
        verbose=not args.quiet,
        generate_report=not args.no_report,
        report_format=args.report_format,
    )

    runner = CheckRunner(config)
    result = runner.run()

    print("\n" + "=" * 60)
    print("Query Difficulty Check Results")
    print("=" * 60)
    print(f"  Personas checked: {result.total_personas}")
    print(f"  Queries checked: {result.total_queries}")
    print(f"  Model correct: {result.total_correct}")
    print(f"  Correct rate: {result.overall_correct_rate * 100:.1f}%")
    print()
    print("  Higher correct rate = questions are too easy (not memory-dependent).")

    if args.regenerate:
        print()
        print(f"  Regeneration attempted: {result.regeneration_attempted}")
        print(f"  Regeneration success: {result.regeneration_success}")
        print(f"  Regeneration failed: {result.regeneration_failed}")

    print()
    print(f"  Duration: {result.duration_seconds:.2f}s")

    if result.report_path:
        print(f"  Markdown report: {result.report_path}")
    if result.json_report_path:
        print(f"  JSON report: {result.json_report_path}")
    if result.enhanced_queries_path:
        print(f"  Enhanced queries: {result.enhanced_queries_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
