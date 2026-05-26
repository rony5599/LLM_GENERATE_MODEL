# Academic Resource Model (Production-Ready Starter)

This project trains a **text classification model** from your folder structure like:

- `academic/class_9_10/science/physics.txt`
- `academic/hsc/commerce/accounting.txt`

The model predicts labels in this form:
`<class_group>__<stream>__<subject>`

---

## 1) Architecture

1. **Data ingestion**: parse `.txt` files and labels from path.
2. **Training**: fine-tune `distilbert-base-uncased` for multi-class classification.
3. **Evaluation**: weighted F1 + accuracy.
4. **Packaging**: save model + tokenizer + metadata to `artifacts/`.
5. **Serving**: FastAPI `/predict` endpoint.
6. **Deployment**: Docker container.

---

## 2) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
```

---

## 3) Data preparation rules

- Keep files as UTF-8 `.txt`.
- Minimum layout: `<class_group>/<stream>/<subject>.txt`
- Add more documents per class for better generalization.
- For production quality, target at least **100+ samples/class**.

---

## 4) Train

```bash
./scripts/train.sh
```

Outputs:
- `artifacts/model/` → model + tokenizer
- `artifacts/metadata.json` → labels + metrics
- `artifacts/training_catalog.csv` → training catalog

---

## 5) Tune

Edit `configs/train_config.yaml`:

- `epochs`: 3–10
- `learning_rate`: try `1e-5`, `2e-5`, `3e-5`, `5e-5`
- `max_length`: 128/256/512
- `warmup_ratio`: 0.05–0.2
- batch sizes according to GPU RAM

Recommended process:
1. Start with 3 epochs and LR `2e-5`.
2. Compare weighted F1.
3. Increase epochs only if validation keeps improving.
4. Stop if overfitting (eval F1 drops while train improves).

---

## 6) Run API locally

```bash
./scripts/serve.sh
```

Test:

```bash
curl -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"text":"Newton laws of motion and acceleration"}'
```

---

## 7) Docker deployment

Build image:

```bash
docker build -t academic-model:1.0.0 .
```

Run container:

```bash
docker run -p 8000:8000 academic-model:1.0.0
```

---

## 8) Production hardening checklist

- Add CI: lint, unit tests, smoke inference test.
- Track experiments (MLflow/W&B).
- Add data versioning (DVC or object-store version tags).
- Add model registry and staged rollout.
- Add monitoring: latency, error rate, confidence drift.
- Add retraining schedule (weekly/monthly).
- Add fallback + input validation for empty/garbage text.

---

## 9) Next upgrades

- Multi-label prediction (if one document maps to multiple subjects).
- Add language-specific models (Bangla/Multilingual BERT).
- Convert to ONNX/TensorRT for faster inference.
- Add RBAC/auth before public API exposure.
