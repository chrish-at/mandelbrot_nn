# Mandelbrot Neural Networks

Neural networks that learn to predict the smooth escape-time function of the Mandelbrot set from raw (x, y) coordinates. Includes scaling law experiments (data, model size, compute) following Kaplan et al. (2020).

## Models

- **MLPRes** -- Baseline deep residual MLP on raw coordinates
- **MLPFourierRes** -- Multi-scale Gaussian Fourier features + residual MLP
- **MLPGatedRes** -- Residual MLP with gated blocks (bilinear, SwiGLU, etc.)
- **MLPFourierGatedRes** -- Fourier features + gated residual backbone

## Project structure

| File | Description |
|------|-------------|
| `data.py` | Escape-time computation, boundary-biased dataset generation (smooth and discrete targets) |
| `models.py` | Model architectures |
| `train.py` | Training script with model selection and checkpointing |
| `scaling.py` | Scaling law sweeps over data size, model size, and compute |
| `render.py` | Renders model predictions, ground truth comparisons, error maps, and zoom views |
| `render_discrete.py` | Binary and smooth Mandelbrot visualizations |
| `viz_dataset.py` | Scatter and density plots of training data |

## Usage

```bash
pip install -r requirements.txt

# Train models
python train.py --model fourier
python train.py --model gated --gate_type bilinear

# Run scaling experiments
python scaling.py                  # full sweep
python scaling.py --axis data      # data scaling only
python scaling.py --plot-only      # regenerate plots from saved results

# Render predictions
python render.py
```

## Requirements

- Python 3.10+
- PyTorch >= 2.7
- numpy, matplotlib, tqdm
