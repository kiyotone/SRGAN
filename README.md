# SRGAN - Super-Resolution Generative Adversarial Network

## Overview
This project implements a Super-Resolution Generative Adversarial Network (SRGAN) using PyTorch. SRGAN is used to generate high-resolution (HR) images from low-resolution (LR) images, enhancing image quality with perceptual and adversarial losses.

## File Structure
```
SRGAN_Project/
│── models/
│   │── __init__.py
│   │── discriminator.pth
│   │── discriminator.py
│   │── generator.pth
│   │── generator.py
│
│── out/
│
│── utils/
│
│── .gitignore
│── dataset.py
│── loss.py
│── requirements.txt
│── train.ipynb
```

## Requirements
To install the necessary dependencies, run:
```
pip install -r requirements.txt
```

## Training
Run the train.ipynb notebook to train the SRGAN model:

## Models
- **Generator**: Generates high-resolution images from low-resolution inputs.
- **Discriminator**: Differentiates between real high-resolution images and generated images.

## Dataset
Modify `dataset.py` to load your custom dataset.

## Output
Generated images and model checkpoints are saved in the `out/` directory.

## Contact
For any questions, feel free to reach out!
