'''
How to run the command (an example):
python data_preprocessing.py --input_dir path/to/input --output_dir path/to/output --workers 4 --datatype uvi
'''

import numpy as np
import os
from PIL import Image
import concurrent.futures
from torchvision import transforms
import argparse

def process_image(args):
    image_path, output_dir, augmentations, datatype = args
    image = Image.open(image_path).convert('L')
    width, height = image.size
    original_img_name = os.path.splitext(os.path.basename(image_path))[0]
    slice_size = 128
    step_size = 64
    grids_found = 0
    images_to_be_saved = 0
    
    for y in range(0, height - slice_size + 1, step_size):
        for x in range(0, width - slice_size + 1, step_size):
            box = (x, y, x + slice_size, y + slice_size)
            slice_img = image.crop(box)
            slice_np = np.array(slice_img)
            if np.sum(slice_np == 0) < 100:
                grids_found += 1
                if datatype == 'uvi':
                    # Apply augmentations only if datatype is 'uvi'
                    for i, augment in enumerate(augmentations):
                        augmented_img = augment(slice_img)
                        file_name = f"{original_img_name}_x{x}_y{y}_aug{i}.png"
                        augmented_img.save(os.path.join(output_dir, file_name))
                        images_to_be_saved += 1
                else:
                    # Save original image if datatype is 'lir'
                    file_name = f"{original_img_name}_x{x}_y{y}.png"
                    slice_img.save(os.path.join(output_dir, file_name))
                    images_to_be_saved += 1
    return grids_found, images_to_be_saved

def process_images_in_parallel(input_dir, output_dir, augmentations, max_workers, datatype):
    image_paths = [os.path.join(input_dir, name) for name in os.listdir(input_dir)]
    total_grids = 0
    total_images = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = [(path, output_dir, augmentations, datatype) for path in image_paths]
        results = executor.map(process_image, tasks)
        for result in results:
            grids, images = result
            total_grids += grids
            total_images += images
    print(f"Total grids found: {total_grids}")
    print(f"Total images to be saved after augmentations: {total_images}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images for data augmentation.")
    parser.add_argument('-i', '--input_dir', required=True, help="Directory containing images.")
    parser.add_argument('-o', '--output_dir', required=True, help="Directory where augmented images will be saved.")
    parser.add_argument('-w', '--workers', type=int, default=4, help="Number of worker processes to use.")
    parser.add_argument('-d', '--datatype', required=True, choices=['uvi', 'lir'], help="Type of data processing: 'uvi' for applying augmentations, 'lir' for no augmentations.")

    args = parser.parse_args()

    augmentations = [
        transforms.RandomVerticalFlip(p=1.0),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
        transforms.ColorJitter(brightness=0.5)
    ]

    process_images_in_parallel(args.input_dir, args.output_dir, augmentations, args.workers, args.datatype)
