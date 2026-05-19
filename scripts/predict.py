import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import cv2
import numpy as np

from serving.inference import InferenceEngine
from utils.visualization import overlay_prediction


def main():
    parser = argparse.ArgumentParser(description="Banana Shelf-Life Prediction")
    parser.add_argument("image_path", type=str, help="Path to banana image")
    parser.add_argument("--model", choices=["ensemble", "cnn", "vit", "hybrid"],
                        default="ensemble", help="Model type")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path for annotated image")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Show result in window")
    parser.add_argument("--gradcam", action="store_true",
                        help="Show Grad-CAM overlay")
    args = parser.parse_args()

    engine = InferenceEngine(model_type=args.model)
    result = engine.predict_from_file(args.image_path)

    print(f"\n{'='*50}")
    print(f"  BANANA SHELF-LIFE PREDICTION")
    print(f"{'='*50}")
    print(f"  File:         {result.get('filename', args.image_path)}")
    print(f"  Shelf Life:   {result['shelf_life_days']} days remaining")
    print(f"  Ripeness:     {result['ripeness_label']}")
    print(f"  Inference:    {result.get('inference_time_ms', 'N/A')} ms")

    if "total_uncertainty" in result:
        ci = result.get("confidence_95_ci", (0, 0))
        print(f"  95% CI:       [{ci[0]:.2f}, {ci[1]:.2f}] days")
        print(f"  Uncertainty:  ±{result['total_uncertainty']:.2f} days")

    if "component_predictions" in result:
        print(f"\n  Component Predictions:")
        for name, val in result["component_predictions"].items():
            print(f"    {name:15s}: {val:.2f} days")

    print(f"{'='*50}")

    if args.visualize or args.output:
        with open(args.image_path, "rb") as f:
            raw = bytearray(f.read())
        arr = np.asarray(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        ci = result.get("confidence_95_ci", None)
        confidence_str = f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci else None

        annotated = overlay_prediction(image, result["shelf_life_days"],
                                        result["ripeness_label"], confidence_str)

        if args.gradcam and "gradcam_overlay" in result:
            overlay_img = result["gradcam_overlay"]
            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
            annotated = np.hstack([annotated, overlay_img])

        if args.output:
            cv2.imwrite(args.output, annotated)
            print(f"\nAnnotated image saved: {args.output}")

        if args.visualize:
            cv2.imshow("Banana Shelf-Life Prediction", annotated)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
