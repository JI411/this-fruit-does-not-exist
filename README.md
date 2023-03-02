# This fruit doesn't exist - synthetic fruit generation

This repository contains code for generating synthetic fruits dataset with Stable Diffusion and Unsupervised Segmentation.
For further details, please refer to [this article](https://wandb.ai/lekomtsev/this-fruit-does-not-exist/reports/Experimets-Report--VmlldzozNjQyNDg2).

## How to use
<a href="https://colab.research.google.com/drive/1jJcucFWIKTuGIbEeJIgX8wkbiPCVlO26" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


Install dependencies:
```bash
git clone https://github.com/JI411/this-fruit-does-not-exist.git
cd this-fruit-does-not-exist
pip install -r requirements.txt
```

Generate dataset:
```bash
python run.py --skip-training --no-seed
```

Train model:
```bash
python run.py --skip-generation --batch=16
```
You also can use any arguments from pytorch-lightning trainer.


