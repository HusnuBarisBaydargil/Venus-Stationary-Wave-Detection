# Stationary Wave Detection and Localization
This repository contains the code for our paper, titled "Detection and Localization of Stationary Waves on Venus Using a Self-Supervised Anomaly Detection Model".

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

After the setup is complete, prepare the dataset using the preprocessing step below:
```bash
python data_preprocessing.py --input_dir path/to/input --output_dir path/to/output --workers 4 --datatype uvi_or_lir
```

## Training 

## Testing

## Citation

## Acknowledgements
