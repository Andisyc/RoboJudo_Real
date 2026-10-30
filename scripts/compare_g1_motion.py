"""Read-only G1 segment comparison, runnable without importing deployment dependencies."""

import argparse
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--segment", type=int)
    parser.add_argument("--reference-fps", type=float, required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--context", type=float, default=2.0)
    parser.add_argument("--max-gap-ms", type=float, default=100.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--model-file", type=Path, help="Optional local copy of deployed ONNX, for body-name metadata only"
    )
    args = parser.parse_args()
    # Importing robojudo itself initializes the deployment/GPU stack; offline analysis must not do that.
    path = Path(__file__).resolve().parents[1] / "robojudo/tools/motion_compare.py"
    spec = importlib.util.spec_from_file_location("motion_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        run = module.load_run(args.run_dir)
        selection = module.select_segment(run, args.segment, args.reference_fps, args.start, args.end, args.context)
        data = module.align_signals(run, selection, args.max_gap_ms, args.model_file)
        comparisons = module.compute_comparisons(data)
        output = args.output_dir or args.run_dir / f"comparison_s{selection['segment']}_{args.start:g}_{args.end:g}"
        report = module.write_report(run, selection, data, comparisons, output, args.max_gap_ms)
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(f"Report: {report.resolve()}\nAligned data: {(output / 'aligned.npz').resolve()}")


if __name__ == "__main__":
    main()
