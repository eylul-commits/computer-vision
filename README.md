
## 1. Install Dependencies (≈2 minutes)

```bash
pip install -r requirements.txt
```

Make sure you are using Python 3.9+ in a virtual environment if possible.

---

## 2. Prepare / Verify Datasets (≈1 minute)

Place your datasets in the expected locations:

- **Fire dataset (YOLO detection, used as classification):**
  - `fire-dataset/archive/data/`
  - Expected structure:

```text
fire-dataset/
└── archive/
    └── data/
        ├── train/
        │   ├── images/*.jpg
        │   └── labels/*.txt
        ├── val/
        │   ├── images/*.jpg
        │   └── labels/*.txt
        └── test/
            ├── images/*.jpg
            └── labels/*.txt
```

- **Smoke dataset (image classification):**

```text
smoke-dataset/
├── train/
│   ├── cloud/*.jpg
│   ├── other/*.jpg
│   └── smoke/*.jpg
├── val/
│   ├── cloud/*.jpg
│   ├── other/*.jpg
│   └── smoke/*.jpg
└── test/
    ├── cloud/*.jpg
    ├── other/*.jpg
    └── smoke/*.jpg
```

---

## 3. Train Your First Model (≈2 minutes setup + training time)

All training is done via `train.py`. During training, the script:

- Uses **train** split for optimization
- Uses **val** split for early stopping / best model selection
- After training, automatically evaluates on the **test** split and saves metrics

### Example Usage: Quick Baseline Test (~10 minutes training)

```bash
python train.py \
--model-type baseline \
    --model-name simple \
    --dataset smoke \
    --epochs 10 \
    --batch-size 32
```
