import torch
import torch.onnx
from pathlib import Path
import numpy as np

from config import cfg
from models.factory import ModelFactory


def export_to_onnx(
    model_type: str = "cnn",
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    input_shape: tuple = (1, 3, 224, 224),
) -> str:
    device = "cpu"
    model = ModelFactory.create(model_type, {
        "backbone": cfg.get("models", "cnn", "backbone", default="efficientnet_v2_s"),
        "dropout": 0.0,
        "embedding_dim": 128,
    })
    model.eval()

    if model_path:
        state = torch.load(model_path, map_location=device, weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

    dummy = torch.randn(input_shape)

    if output_path is None:
        models_dir = Path(cfg.get("paths", "models_dir", default="models/saved"))
        output_path = str(models_dir / f"{model_type}_model.onnx")

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["shelf_life", "stage_logits", "log_var", "embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "shelf_life": {0: "batch_size"},
            "stage_logits": {0: "batch_size"},
        },
    )

    return output_path


if __name__ == "__main__":
    path = export_to_onnx()
    print(f"Model exported to: {path}")
