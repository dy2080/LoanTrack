"""
================================================================================
cli.py — Command-line interface: train · predict · serve
================================================================================

Usage:
    python cli.py train               Train all models (with & without log)
    python cli.py train --log         Train with log transform only
    python cli.py predict             Run demo inference on sample applicant
    python cli.py serve               Start the FastAPI web server
    python cli.py serve --port 5000   Start on a custom port
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import DEFAULT_MODEL


def cmd_train(args):
    """Train pipeline."""
    from train import run_pipeline

    if args.both:
        from train import main as train_main
        train_main()
    else:
        tag = "with_log" if args.log else "no_log"
        run_pipeline(use_log=args.log, tag=tag)


def cmd_predict(args):
    """Run demo inference."""
    from src.loan_ml.inference.predictor import demo_inference

    model_file = args.model or DEFAULT_MODEL
    demo_inference(model_filename=model_file, apply_log=args.log)


def cmd_serve(args):
    """Launch FastAPI dev server."""
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Loan Eligibility ML — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── train ──
    p_train = sub.add_parser("train", help="Train all ML models")
    p_train.add_argument("--log", action="store_true", default=True,
                         help="Apply log transformation (default: True)")
    p_train.add_argument("--no-log", dest="log", action="store_false",
                         help="Skip log transformation")
    p_train.add_argument("--both", action="store_true",
                         help="Run both with and without log (full run)")

    # ── predict ──
    p_pred = sub.add_parser("predict", help="Run demo inference")
    p_pred.add_argument("--model", type=str, default=None,
                        help="Path to .pkl model file")
    p_pred.add_argument("--log", action="store_true", default=True)
    p_pred.add_argument("--no-log", dest="log", action="store_false")

    # ── serve ──
    p_serve = sub.add_parser("serve", help="Start the web server")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
