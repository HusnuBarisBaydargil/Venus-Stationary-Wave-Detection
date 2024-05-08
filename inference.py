import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import argparse
from vqgan import VQGAN  # Make sure this import is correct for your setup

def get_args():
    parser = argparse.ArgumentParser(description="Run inference on a large image using a trained VQGAN model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the trained model weights.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output images and logs.")
    parser.add_argument("--device", type=str, default="cuda:2", help="Compute device to use (default: cuda:2).")
    return parser.parse_args()

def load_image(path):
    return Image.open(path)

def sliding_window(image, step_size, window_size):
    for y in range(0, image.size[1] - window_size[1] + 1, step_size):
        for x in range(0, image.size[0] - window_size[0] + 1, step_size):
            yield (x, y, image.crop((x, y, x + window_size[0], y + window_size[1])))

def preprocess_image(image):
    image = image.convert('L')
    if np.sum(np.array(image) == 0) > 100:
        return None
    image_tensor = transforms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
    return image_tensor

def load_model(weights_path, device):
    model = VQGAN()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def infer_and_draw(args):
    image = load_image(args.image_path)
    draw = ImageDraw.Draw(image)
    model = load_model(args.weights_path, args.device)
    losses = []
    log_entries = []

    for (x, y, window) in sliding_window(image, step_size=64, window_size=(128, 128)):
        processed = preprocess_image(window)
        if processed is not None:
            processed = processed.to(args.device)
            with torch.no_grad():
                output = model(processed)
                reconstructed_image = output[0] if isinstance(output, tuple) else output
                loss = torch.nn.functional.l1_loss(processed, reconstructed_image).item()
                losses.append((x, y, loss))
                log_entries.append(f"Grid: Loss: {loss:.5f}, Coordinates: ({x}, {y}) to ({x + 128}, {y + 128})")

    mean_loss = np.mean([loss[2] for loss in losses])
    std_dev_loss = np.std([loss[2] for loss in losses])
    log_entries.append(f"Mean L1 Loss: {mean_loss:.5f}")
    log_entries.append(f"Standard Deviation of L1 Loss: {std_dev_loss:.5f}")

    with open(os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.image_path))[0] + "_logs.txt"), 'w') as f:
        for entry in log_entries:
            f.write(entry + '\n')

    image.save(os.path.join(args.output_dir, os.path.basename(args.image_path).replace('.png', '_annotated.png')))
    return losses, mean_loss, std_dev_loss

if __name__ == "__main__":
    args = get_args()
    losses, mean_loss, std_dev_loss = infer_and_draw(args)
    print(f"Mean L1 Loss: {mean_loss:.5f}, Standard Deviation of L1 Loss: {std_dev_loss:.5f}")
