# Multilingual Transliteration System

Convert romanized text to native scripts for Hindi, Bengali, and Tamil.

## Setup

1. Install Python 3.8 or higher
2. Run setup:
```
setup.bat
```

## Usage

### Option 1: Run Everything
```
run_all.bat
```

### Option 2: Run Step by Step

Activate environment first:
```
venv\Scripts\activate.bat
```

Then run in order:

1. Prepare data:
```
python src\prepare_data.py
```

2. Train model (2-4 hours):
```
python src\train.py
```

3. Optimize with CTranslate2:
```
python src\optimize.py
```

4. Evaluate:
```
python src\evaluate.py
```

5. Launch app:
```
python deployment\app.py
```
Open http://localhost:7860 in browser

## Project Structure

```
transliteration/
├── src/
│   ├── prepare_data.py    # Download and prepare dataset
│   ├── train.py           # Train mT5 model
│   ├── optimize.py        # Convert to CTranslate2
│   └── evaluate.py        # Test model performance
├── deployment/
│   └── app.py             # Gradio web interface
├── requirements.txt       # Python packages
├── setup.bat              # Setup script
└── run_all.bat            # Run complete pipeline
```

## What It Does

1. **Data Preparation**: Downloads Aksharantar dataset for Hindi, Bengali, Tamil
2. **Training**: Trains mT5 model on transliteration task
3. **Optimization**: Converts to CTranslate2 for faster inference
4. **Evaluation**: Tests accuracy and speed
5. **Deployment**: Creates web interface

## Expected Results

- Accuracy: 90-94%
- Speed improvement: 3-5x faster with CTranslate2
- Model size: Reduced by 70%

## Examples

Hindi: `namaste` → नमस्ते
Bengali: `nomoskar` → নমস্কার
Tamil: `vanakkam` → வணக்கம்

## Requirements

- Windows 10/11
- Python 3.8+
- 16GB RAM recommended
- GPU optional (faster training)

## Deploy to HuggingFace Spaces

1. Create account at https://huggingface.co
2. Create new Space with Gradio SDK
3. Upload files:
   - deployment/app.py
   - requirements.txt
   - models/ folder (after training)
4. Add README.md header:
```yaml
---
title: Transliteration
emoji: 🌐
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
---
```

## Files Generated

After running:
- `data/train.jsonl` - Training data
- `data/test.jsonl` - Test data
- `models/transliteration/` - Trained PyTorch model
- `models/transliteration_ct2/` - Optimized CTranslate2 model
- `benchmark_results.json` - Speed comparison
- `evaluation_results.json` - Accuracy metrics
