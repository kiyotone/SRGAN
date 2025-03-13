SRGAN - Super-Resolution GAN using PyTorch
==========================================

Project Description:
---------------------
This project implements an SRGAN (Super-Resolution Generative Adversarial Network) using PyTorch. The model enhances low-resolution images to high-resolution ones by training a generator and discriminator in an adversarial framework.

File Structure:
---------------
SRGAN/
│── models/                  # Directory containing model implementations
│   │── __init__.py          # Python init file
│   │── discriminator.py     # Discriminator model architecture
│   │── generator.py         # Generator model architecture
│   │── discriminator.pth    # Pre-trained discriminator weights (if available)
│   │── generator.pth        # Pre-trained generator weights (if available)
│
│── out/                     # Directory for storing output images
│── utils/                   # Utility functions (e.g., image processing)
│── dataset.py               # Dataset loading and preprocessing
│── loss.py                  # Loss functions used in training (e.g., MSE, Perceptual Loss)
│── train.ipynb              # Jupyter Notebook for training and evaluation
│── requirements.txt         # Python dependencies
│── .gitignore               # Files to ignore in version control

Setup Instructions:
-------------------
1. Clone the repository:


Setup Instructions:
-------------------
1. Clone the repository:
   ```
   git clone https://github.com/your-repo/srgan.git
   cd srgan
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  (Linux/macOS)
   venv\Scripts\activate     (Windows)
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
