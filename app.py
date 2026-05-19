import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import io
import base64
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json

from serving.inference import InferenceEngine


engine: Optional[InferenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    try:
        engine = InferenceEngine(model_type="ensemble")
        print(f"Model loaded: {engine.metadata}")
    except Exception as e:
        print(f"Warning: {e}")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Banana Shelf-Life Prediction API",
    description="AI-powered banana freshness prediction with uncertainty quantification",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Banana Shelf-Life Predictor v2</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
               background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               min-height: 100vh; padding: 20px; display: flex; justify-content: center; align-items: flex-start; }
        .card { background: rgba(255,255,255,0.95); border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px; max-width: 800px; width: 100%; margin: 20px 0; backdrop-filter: blur(10px); }
        h1 { font-size: 32px; background: linear-gradient(135deg, #667eea, #764ba2);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 4px; }
        .subtitle { color: #666; margin-bottom: 24px; font-size: 15px; }
        .upload-area { border: 2px dashed #667eea; border-radius: 16px; padding: 40px; text-align: center;
                       cursor: pointer; transition: all 0.3s; background: #f8f9ff; }
        .upload-area:hover { border-color: #764ba2; background: #f0f0ff; transform: translateY(-2px);
                             box-shadow: 0 8px 25px rgba(102,126,234,0.2); }
        .upload-area img { max-width: 100%; max-height: 350px; border-radius: 12px; margin-top: 16px;
                           box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        .predict-btn { background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none;
                       padding: 16px 48px; border-radius: 12px; font-size: 17px; font-weight: 600;
                       cursor: pointer; margin-top: 20px; transition: all 0.3s;
                       box-shadow: 0 4px 15px rgba(102,126,234,0.4); }
        .predict-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102,126,234,0.5); }
        .predict-btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .result { margin-top: 24px; padding: 24px; border-radius: 16px; display: none; }
        .result.visible { display: block; animation: fadeIn 0.5s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .result.fresh { background: linear-gradient(135deg, #e8f5e9, #c8e6c9); border: 1px solid #a5d6a7; }
        .result.warning { background: linear-gradient(135deg, #fff3e0, #ffe0b2); border: 1px solid #ffcc80; }
        .result.spoiled { background: linear-gradient(135deg, #ffebee, #ffcdd2); border: 1px solid #ef9a9a; }
        .result-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }
        .metric-card { background: rgba(255,255,255,0.8); padding: 16px; border-radius: 12px;
                       backdrop-filter: blur(5px); }
        .metric-card .label { font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
        .metric-card .value { font-size: 26px; font-weight: bold; margin-top: 4px; }
        .confidence-bar { height: 8px; border-radius: 4px; background: #e0e0e0; margin-top: 8px; overflow: hidden; }
        .confidence-bar .fill { height: 100%; border-radius: 4px; transition: width 1s ease; }
        .component-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
                          gap: 8px; margin-top: 12px; }
        .component-item { background: rgba(255,255,255,0.6); padding: 8px 12px; border-radius: 8px; font-size: 13px; }
        .spinner { margin: 20px auto; border: 4px solid #f0f0ff; border-top: 4px solid #667eea;
                   border-radius: 50%; width: 48px; height: 48px; animation: spin 0.8s linear infinite;
                   display: none; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .error { color: #d32f2f; padding: 16px; border-radius: 12px; margin-top: 16px; display: none;
                  background: #ffebee; }
        input[type="file"] { display: none; }
        .model-select { margin: 16px 0; display: flex; gap: 8px; justify-content: center; }
        .model-btn { padding: 8px 20px; border-radius: 20px; border: 2px solid #667eea; background: white;
                     cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.3s; }
        .model-btn.active { background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-color: transparent; }
    </style>
</head>
<body>
<div class="card">
    <h1>Banana Shelf-Life Predictor v2</h1>
    <p class="subtitle">Upload a banana photo — AI predicts freshness with confidence intervals &amp; uncertainty</p>

    <div class="model-select" id="modelSelect">
        <button class="model-btn active" data-model="ensemble">Ensemble</button>
        <button class="model-btn" data-model="cnn">CNN</button>
        <button class="model-btn" data-model="hybrid">Hybrid</button>
    </div>

    <div class="upload-area" id="uploadArea">
        <p id="uploadText">Click to upload or drag & drop a banana image</p>
        <img id="preview" style="display:none;" />
    </div>
    <input type="file" id="fileInput" accept="image/*" />

    <div style="text-align:center;">
        <button class="predict-btn" id="predictBtn" disabled onclick="predict()">Predict Shelf Life</button>
    </div>
    <div class="spinner" id="spinner"></div>
    <div id="error" class="error"></div>
    <div class="result" id="result">
        <h3 id="resultTitle">Prediction Result</h3>
        <div class="result-grid" id="resultGrid"></div>
        <div id="componentSection" style="display:none; margin-top:16px;">
            <h4 style="margin-bottom:8px;">Model Components</h4>
            <div class="component-grid" id="componentGrid"></div>
        </div>
    </div>
</div>

<script>
    let selectedModel = 'ensemble';

    document.querySelectorAll('.model-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedModel = btn.dataset.model;
        });
    });

    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const preview = document.getElementById('preview');
    const uploadText = document.getElementById('uploadText');
    const predictBtn = document.getElementById('predictBtn');

    uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
    uploadArea.addEventListener('drop', e => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; handleFile(e.dataTransfer.files[0]); }
    });
    fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) return;
        const reader = new FileReader();
        reader.onload = e => { preview.src = e.target.result; preview.style.display = 'block'; uploadText.style.display = 'none'; predictBtn.disabled = false; };
        reader.readAsDataURL(file);
    }

    async function predict() {
        if (!fileInput.files.length) return;
        const btn = document.getElementById('predictBtn');
        const spinner = document.getElementById('spinner');
        const resultDiv = document.getElementById('result');
        const errorDiv = document.getElementById('error');

        btn.disabled = true;
        spinner.style.display = 'block';
        resultDiv.classList.remove('visible');
        resultDiv.style.display = 'none';
        errorDiv.style.display = 'none';

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('model', selectedModel);

        try {
            const resp = await fetch('/predict', { method: 'POST', body: formData });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || 'Prediction failed');

            const stageClass = data.shelf_life_days > 5 ? 'fresh' : data.shelf_life_days > 1 ? 'warning' : 'spoiled';
            const ci = data.confidence_95_ci || [0, 0];
            const uncertainty = data.total_uncertainty || 0;
            const confidencePct = Math.max(0, Math.min(100, (1 - uncertainty / 5) * 100));

            document.getElementById('resultGrid').innerHTML = `
                <div class="metric-card"><div class="label">Shelf Life</div><div class="value" style="color:${stageClass === 'fresh' ? '#2e7d32' : stageClass === 'warning' ? '#e65100' : '#c62828'}">${data.shelf_life_days} days</div></div>
                <div class="metric-card"><div class="label">Ripeness Stage</div><div class="value" style="font-size:18px">${data.ripeness_label}</div></div>
                <div class="metric-card"><div class="label">95% Confidence Interval</div><div class="value" style="font-size:18px">[${ci[0]}, ${ci[1]}] days</div></div>
                <div class="metric-card"><div class="label">Confidence Score</div><div class="value" style="font-size:20px">${confidencePct.toFixed(0)}%</div>
                    <div class="confidence-bar"><div class="fill" style="width:${confidencePct}%; background: linear-gradient(90deg, #4caf50, #8bc34a);"></div></div>
                </div>
            `;

            const compSection = document.getElementById('componentSection');
            const compGrid = document.getElementById('componentGrid');
            if (data.component_predictions) {
                compSection.style.display = 'block';
                compGrid.innerHTML = Object.entries(data.component_predictions).map(([k,v]) =>
                    `<div class="component-item"><strong>${k}</strong>: ${v.toFixed(2)}d</div>`
                ).join('');
            } else {
                compSection.style.display = 'none';
            }

            resultDiv.className = 'result ' + stageClass + ' visible';
            resultDiv.style.display = 'block';
        } catch (err) {
            errorDiv.textContent = 'Error: ' + err.message;
            errorDiv.style.display = 'block';
        } finally {
            btn.disabled = false;
            spinner.style.display = 'none';
        }
    }
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_PAGE


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": engine is not None,
        "metadata": engine.metadata if engine else {},
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = "ensemble"):
    global engine

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    if engine is None or engine.model_type != model:
        try:
            engine = InferenceEngine(model_type=model)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {e}")

    try:
        result = engine.predict(image)
        result["filename"] = file.filename

        ci = result.pop("confidence_95_ci", None)
        if ci and isinstance(ci, tuple) and len(ci) == 2:
            result["confidence_95_ci"] = [round(float(ci[0]), 2), round(float(ci[1]), 2)]

        overlay = result.pop("gradcam_overlay", None)
        if overlay is not None:
            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            result["gradcam_b64"] = base64.b64encode(buffer).decode("utf-8")

        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...), model: str = "ensemble"):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is not None:
            result = engine.predict(image)
            result["filename"] = file.filename
            results.append(result)
    return JSONResponse(results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
