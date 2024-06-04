# Stationary Wave Detection and Localization
This repository contains the code for our paper, titled "Detection and Localization of Stationary Waves on Venus Using a Self-Supervised Anomaly Detection Model".
![](images/StationaryWave_Example_LIR_UVI.png)
## System Requirements and Setup
We run our experiments in Ubuntu 20.04 using Nvidia A100 GPU.
To set up and use the required environment for this project, follow these steps:

1. **Create the Conda environment:**
   ```bash
   conda create -n venus_waves python=3.9

2. **Activate the environment:**
   ```bash
   conda activate venus_waves

3. **Install the required libraries:**
   ```bash
   pip install -r requirements.txt

## Download the UVI and LIR datasets

[Download from Google Drive](https://drive.google.com/uc?export=download&id=1T4ZFRP7V-_1gKvfZNUgg-xHw8OmGsbXw)

## Training 
Prepare the dataset using the preprocessing step below:
```bash
python data_preprocessing.py --input_dir path/to/input --output_dir path/to/output --workers 4 --datatype uvi_or_lir
```

If you want to crop your grids from images, use our `grid_cropper_UI.py` script.

To train the model, run the script `train.py`

## Testing 
To test the model after training, run the script below:
```bash
python inference.py --image_path /path/to/inference/image \
                    --weights_path /path/to/model/weight \
                    --output_dir /path/to/resulting/image \
                    --device cuda:2
```
You can download our pretrained weight [from here](https://drive.google.com/file/d/1gfI0BjzUdce8i8qY8tO_pkzfCzzGx8zA/view?usp=sharing)
## Citation
