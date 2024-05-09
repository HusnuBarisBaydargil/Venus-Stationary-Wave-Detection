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

## Training 
Prepare the dataset using the preprocessing step below:
```bash
python data_preprocessing.py --input_dir path/to/input --output_dir path/to/output --workers 4 --datatype uvi_or_lir
```
input_dir: The directory where your dataset is located.
output_dir: The directory to which you want the preprocessed grid images to be saved.
workers: Number of CPU cores you would like to utilize. WARNING: Increasing this number can significantly slow down your computer. To check the available number of CPU cores in your Ubuntu, you can type "lscpu" to the terminal, and type "wmic cpu get NumberOfCores, NumberOfLogicalProcessors" in cmd in Windows.
datatype: The data type you're preprocessing. Either uvi or lir.


## Testing

## Citation

## Acknowledgements
