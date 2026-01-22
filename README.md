# Synapse Challenge: High-Precision sEMG Gesture Recognition
![694640b1e0efc_synapse-the-neurotech-challenge](https://github.com/user-attachments/assets/5ebe4056-caa3-4aaa-bea7-f0d5ff9d3184)

## 1. Project Overview
This project implements a robust deep learning pipeline for classifying 8-channel surface Electromyography (sEMG) signals into 5 distinct hand gestures. The solution achieves a top-1 accuracy of 78.48% on the hidden test set (Subjects 21-25) by utilizing a specialized ensemble architecture and Test Time Augmentation (TTA).

The core challenge addressed in this solution is the "Inter-Subject Variability" problem, where muscle signals differ significantly between users. Our approach mitigates this using a dual-stream training strategy that balances signal precision with generalization.

## 2. Methodology & Approach

### 2.1 The "Dual-Specialist" Ensemble
Instead of relying on a single model, we trained two distinct "specialist" variants of the same architecture to capture different aspects of the signal features:

1.  **The Precision Model (Standard):** Trained on raw, clean signal data. This model specializes in distinct, high-amplitude gestures (Gesture 0 and Gesture 4).
2.  **The Generalization Model (MixUp):** Trained using MixUp regularization (alpha=0.4). This model specializes in disambiguating confused classes (specifically Gesture 2 vs. Gesture 3) by learning linear interpolations of the feature space.

During inference, predictions are generated via a weighted soft-voting mechanism:
* Weight Standard: 0.45
* Weight MixUp: 0.55

### 2.2 Test Time Augmentation (TTA)
To further stabilize predictions against sensor noise and shift artifacts, we implement TTA. For every input signal, the model predicts on a batch of 5 variations:
1.  Original Signal
2.  Shift Left (10 time steps)
3.  Shift Right (10 time steps)
4.  Gaussian Noise Injection (scale=0.02)
5.  Amplitude Scaling (1.05x)

The final probability is the mean of these 5 predictions, significantly reducing variance and "borderline" errors.

## 3. Model Architecture
**Architecture:** SE-ResNet-1D (Squeeze-and-Excitation Residual Network)
**Input Shape:** (Batch, 8, 2560)

The model consists of three main stages:
1.  **Feature Extraction:** A stack of 1D Convolutional layers with Residual connections to prevent vanishing gradients.
2.  **Channel Attention:** Squeeze-and-Excitation (SE) blocks are embedded within residual units. This allows the network to dynamically weight specific sensors (channels) based on relevance (e.g., ignoring the forearm sensor if the bicep sensor contains the primary signal).
3.  **Classification:** Global Average Pooling followed by a linear projection to the 5 target classes.

This architecture was chosen over Transformers to minimize parameter count and inference latency while effectively capturing local temporal patterns (spikes) characteristic of sEMG data.

## 4. Repository Structure

```text
Synapse_Challenge/
├── checkpoints/               # Trained model weights
│   ├── model_standard.pth     # Precision specialist weights
│   └── model_mixup.pth        # Generalization specialist weights
│
├── data/                      # Data directory
│   └── processed/
│       └── synapse_full_dataset.pt  # Pre-processed tensor file
│
├── reports/                   # Output directory for results
│   ├── submission.csv         # Final predictions
│   └── images/                # Performance graphs (Confusion Matrix, etc.)
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── dataset.py             # Custom PyTorch Dataset class
│   ├── inference.py           # Main execution script for generating predictions
│   └── model.py               # SynapseNet model architecture definition
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 5. Installation & Setup

### Prerequisites

* Python 3.8 or higher
* CUDA-enabled GPU (Recommended for faster inference)

### Installation

1. Navigate to the project root directory.
2. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## 6. How to Run Inference

To generate predictions using the pre-trained models, run the `inference.py` script located in the `src` folder.

### Basic Usage

Run with default paths (assumes folder structure matches Section 4):

```bash
python src/inference.py
```

### Custom Usage

You can specify custom paths for the data, model checkpoints, and output location using command-line arguments:

```bash
python src/inference.py --data /path/to/your/dataset.pt --checkpoints /path/to/weights --output /path/to/save/results
```

**Arguments:**

* `--data`: Path to the input .pt dataset file.
* `--checkpoints`: Directory containing `model_standard.pth` and `model_mixup.pth`.
* `--output`: Directory where `submission.csv` will be saved.

## 7. Performance Results

**Final Evaluation (Subjects 21-25):**

* **Accuracy:** 78.48%
* **Strategy:** Global Weighted Ensemble + TTA

**Class-wise Performance Highlights:**

* Gesture 4: 97% F1-Score (High Robustness)
* Gesture 0: 77% F1-Score (Significantly improved via Ensemble)
* Gesture 2/3 Confusion: Successfully resolved via MixUp training.

