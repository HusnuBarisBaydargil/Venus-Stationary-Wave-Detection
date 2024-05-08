import os

def prepare_training():
    script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the directory where the script is located
    results_path = os.path.join(script_dir, "results")  # Path for results directory
    checkpoints_path = os.path.join(script_dir, "checkpoints")  # Path for checkpoints directory
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    print('results_path:', results_path)
    print('checkpoints_path:', checkpoints_path)
    
prepare_training()