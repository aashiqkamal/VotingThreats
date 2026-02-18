# Analyzing Physical Adversarial Example Threats to Machine Learning in Election Systems

We provide code for attacking a ResNet20 model trained on the Combined voting grayscale dataset link (https://zenodo.org/records/18344123).
All attacks provided here are done on OnlyBubble voting grayscale dataset using PyTorch.

Each attack can be run by uncommenting one of the lines in the main. 

We provide attack code for the APGD-L1, APGD-L2, APGD-Linf, L0-PGD, L0-Linf, L0-Sigma and model architecture files. 

# Step by Step Guide

<ol>
  <li>Install the packages listed in the Software Installation Section (see below).</li>
  <li>Download the model from the Google Drive link listed in the Models Section.</li>
  <li>Download the dataset to attack from the link given in the Dataset section.</li>
  <li>Open the PhysicalAdversarialThreats.py file in the Python IDE of your choice. Choose one of the attack lines and uncomment it. Run the main.</li>
</ol>

# Software Installation 

We use the following software packages: 
<ul>
  <li>python==3.10.18</li>
  <li>pytorch==2.5.1+cu121</li>
  <li>torchvision==0.20.1+cu121</li>
  <li>numpy==2.1.2</li>
  <li>opencv-python== (optional, only needed if you use image save/load utilities)</li>
</ul>

Tested on Unity HPC Jupyter environment (Python 3.10.18, CUDA 12.1)

# Models

We provide the following model:
<ul>
  <li>ResNet20-C model</li>
</ul>

The models can be downloaded [here] https://drive.google.com/file/d/1JZMPdHvKBBjPm0q3VCpsKsLnISQ_PHu0/view?usp=sharing.

# Dataset

We applied the attack on val_OnlyBubbles_Grayscale.pth (validation OnlyBubbles dataset) Link (https://zenodo.org/records/18344123).

# Contact

For questions or concerns please contact the author at: aashiqkamal@uri.edu
