import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import argparse
from vqgan import VQGAN
import random

def get_args():
    parser = argparse.ArgumentParser(description="Run inference on a large image using a trained VQGAN model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the trained model weights.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output images and logs.")
    parser.add_argument("--device", type=str, default="cuda:2", help="Compute device to use (default: cuda:2).")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=128, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 3)')
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

def load_model(weights_path, device, args):
    model = VQGAN(args).to(device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def infer_and_draw(weights_path, image_path, device, output_dir, args):
    image = load_image(image_path)
    draw = ImageDraw.Draw(image)
    model = load_model(weights_path, device, args)
    losses = []
    log_entries = []

    for (x, y, window) in sliding_window(image, step_size=64, window_size=(128, 128)):
        processed = preprocess_image(window)
        if processed is not None:
            processed = processed.to(device)
            with torch.no_grad():
                output = model(processed)
                reconstructed_image = output[0] if isinstance(output, tuple) else output
                loss = torch.nn.functional.l1_loss(processed, reconstructed_image).item()
                losses.append((x, y, loss))
                log_entries.append(f"Grid: Loss: {loss:.5f}, Coordinates: ({x}, {y}) to ({x + 128}, {y + 128})")

    mean_loss = np.mean([loss[2] for loss in losses])
    threshold = mean_loss * 1.2
    high_loss_coords = [(x, y) for x, y, loss in losses if loss > threshold]

    if high_loss_coords:
        avg_x = np.mean([x for x, y in high_loss_coords])
        avg_y = np.mean([y for x, y in high_loss_coords])

        filtered_coords = [(x, y) for x, y in high_loss_coords if abs(x - avg_x) <= 400 and abs(y - avg_y) <= 400]

        if filtered_coords:
            min_x = min(x for x, y in filtered_coords)
            max_x = max(x + 128 for x, y in filtered_coords)
            min_y = min(y for x, y in filtered_coords)
            max_y = max(y + 128 for x, y in filtered_coords)

            draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=6)
            log_entries.append(f"Final Bounding Box: Coordinates: ({min_x}, {min_y}) to ({max_x}, {max_y})")

    log_entries.append(f"Mean L1 Loss: {mean_loss:.5f}")
    log_entries.append(f"Standard Deviation of L1 Loss: {np.std([loss[2] for loss in losses]):.5f}")

    with open(os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_logs.txt"), 'w') as f:
        for entry in log_entries:
            f.write(entry + '\n')

    image.save(os.path.join(output_dir, os.path.basename(image_path).replace('.png', '_annotated.png')))
    return losses, mean_loss, np.std([loss[2] for loss in losses])

if __name__ == "__main__":
    args = get_args()
    losses, mean_loss, std_dev_loss = infer_and_draw(
        args.weights_path,
        args.image_path,
        args.device,
        args.output_dir,
        args
    )
    # print(f"Mean L1 Loss: {mean_loss:.5f}, Standard Deviation of L1 Loss: {std_dev_loss:.5f}")
