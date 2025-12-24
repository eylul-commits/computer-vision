
## 1. Install Dependencies (≈2 minutes)

```bash
pip install -r requirements.txt
```

Make sure you are using Python 3.9+ in a virtual environment if possible.

---

## 2. Prepare / Verify Datasets (≈1 minute)

Datasets:
1. Smoke Detection Dataset (Sage Continuum / UIUC): https://huggingface.co/datasets/sagecontinuum/smokedataset
2. D-Fire Dataset for Smoke and Fire Detection: https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo/data

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

### (Optional) Run the Setup Script

You **do not need** to run `setup.py` every time. It’s a convenience script that can:

- Create common output folders (`models/`, `logs/`, `results/`, `figures/`, `outputs/`)
- Verify your datasets exist at:
  - `fire-dataset/archive/data/`
  - `smoke-dataset/`
- Optionally install dependencies (it will prompt: `Install dependencies? (y/n)`)

Run it once if you want the checks/auto-folder creation:

```bash
python setup.py
```

If you already installed requirements and your dataset paths are correct, you can **skip** this and go straight to training with `train.py`.

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

## 4. Compare Multiple Models (Optional)

Once you’ve trained several models, you can compare them with:

```bash
python evaluate.py --dataset smoke
```

This:

- Loads all `*_config.json` and `*_test_results.json` files from `models/`
- Filters them to the chosen dataset (here: `smoke`)
- Produces:
  - A CSV comparison table: `results/model_comparison.csv`
  - Plots under: `figures/comparison/`

You can delete `--dataset` to compare all experiments across both datasets.

---
