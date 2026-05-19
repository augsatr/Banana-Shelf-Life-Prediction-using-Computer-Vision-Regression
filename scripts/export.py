import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from serving.onnx_export import export_to_onnx


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model", choices=["cnn", "vit"], default="cnn")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    path = export_to_onnx(
        model_type=args.model,
        model_path=args.checkpoint,
        output_path=args.output,
    )
    print(f"Model exported to: {path}")


if __name__ == "__main__":
    main()
