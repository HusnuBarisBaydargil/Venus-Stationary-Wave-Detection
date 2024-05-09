import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from vqgan import VQGAN
from discriminator import Discriminator
from utils import load_data
from lpips import LPIPS
from torchvision import utils as vutils
import numpy as np
from datetime import datetime
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    args = get_args()
    initialize_cuda_devices(args.gpus)
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    trainer = TrainVQGAN(args, local_rank)
    trainer.train()

def initialize_cuda_devices(gpu_ids):
    physical_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    available_gpus = [str(i) for i in range(len(physical_ids))]

    for gpu in gpu_ids.split(','):
        if gpu not in available_gpus:
            print(f"Error: Requested logical GPU {gpu} which maps to physical GPU {physical_ids[int(gpu)]} is not available.")
            exit(1)
        print(f"Logical Device {gpu} is available. Maps to Physical GPU {physical_ids[int(gpu)]}")

def get_args():
    parser = argparse.ArgumentParser(description="Train VQGAN Model")
    parser.add_argument('--gpus', type=str, default='2', help='Comma-separated list of GPU IDs to use (e.g., "0,1")')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
    parser.add_argument('--threshold', default=10000, type=int, help='Threshold value')
    parser.add_argument('--learning_rate', default=2.25e-03, type=float, help='Learning rate')
    parser.add_argument('--epsilon', default=1e-08, type=float, help='Optimizer epsilon')
    parser.add_argument('--beta1', default=0.9, type=float, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', default=0.999, type=float, help='Beta2 for Adam optimizer')
    parser.add_argument('--data_path', default='/path/to/dataset', type=str, help='Dataset path')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--experiment_type', default='UVI', type=str, help='Experiment type')
    args = parser.parse_args()
    return args

class TrainVQGAN():
    def __init__(self, args, local_rank):
        self.args = args
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")        
        self.vqgan = VQGAN(device=self.device)
        self.discriminator = Discriminator().to(self.device)
        self.perceptual_loss = LPIPS().eval().to(self.device)

        self.vqgan = DDP(self.vqgan, device_ids=[local_rank])
        self.discriminator = DDP(self.discriminator, device_ids=[local_rank])

        self.dataset_path = self.args.data_path
        self.epochs = self.args.epochs
        self.threshold = self.args.threshold
        self.opt_vq, self.opt_disc = self.configure_optimizers()
        self.prepare_folders()

    def prepare_folders(self):
        date_today = datetime.now().strftime("%Y%m%d_%H%M")
        base_dir = os.path.join(os.getcwd(), "Experiments", f"{date_today}_{self.args.experiment_type}")
        os.makedirs(base_dir, exist_ok=True)
        print(f"Experiment base directory created at: {base_dir}")
        
        self.results_path = os.path.join(base_dir, "results")
        self.checkpoints_path = os.path.join(base_dir, "checkpoints")
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.checkpoints_path, exist_ok=True)
        
        self.log_file_path = os.path.join(base_dir, "training_log.txt")
        self.log_file = open(self.log_file_path, "a")
        print(f"Logging to {self.log_file_path}")

    def configure_optimizers(self):
        vqgan_model = self.vqgan.module if hasattr(self.vqgan, 'module') else self.vqgan
        opt_vq = torch.optim.Adam(
            list(vqgan_model.encoder.parameters()) +
            list(vqgan_model.decoder.parameters()) +
            list(vqgan_model.codebook.parameters()) +
            list(vqgan_model.quant_conv.parameters()) +
            list(vqgan_model.post_quant_conv.parameters()),
            lr=self.args.learning_rate, eps=self.args.epsilon,
            betas=(self.args.beta1, self.args.beta2))

        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=self.args.learning_rate, eps=self.args.epsilon,
                                    betas=(self.args.beta1, self.args.beta2))
        return opt_vq, opt_disc

    def train(self):
        train_loader, sampler = load_data(self.dataset_path, self.args.batch_size, is_distributed=True, rank=dist.get_rank(), num_replicas=dist.get_world_size())
        steps_per_epoch = len(train_loader)
        global_iteration = 0 

        for epoch in range(self.epochs):
            sampler.set_epoch(epoch)
            with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch", disable=not (dist.get_rank() == 0)) as pbar:
                for i, imgs in enumerate(train_loader):
                    imgs = imgs.to(self.device)
                    decoded_images, _, q_loss = self.vqgan.module(imgs)  
                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)
                    disc_factor = self.vqgan.module.adopt_weight(1., epoch * steps_per_epoch + i, threshold=self.threshold)
                    perceptual_loss = self.perceptual_loss(imgs, decoded_images).mean()
                    rec_loss = torch.abs(imgs - decoded_images).mean()
                    perceptual_rec_loss = perceptual_loss + rec_loss
                    g_loss = -disc_fake.mean()
                    vq_loss = perceptual_rec_loss + q_loss.mean() + disc_factor * g_loss
                    d_loss_real = F.relu(1. - disc_real).mean()
                    d_loss_fake = F.relu(1. + disc_fake).mean()
                    gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)
                    self.opt_disc.zero_grad()
                    gan_loss.backward()
                    self.opt_vq.step()
                    self.opt_disc.step()

                    if i % 100 == 0:
                        with torch.no_grad():
                            real_images = (imgs[:4].add(1) / 2)
                            fake_images = (decoded_images[:4].add(1) / 2)
                            real_fake_images = torch.cat((real_images, fake_images), dim=0)
                            vutils.save_image(real_fake_images, os.path.join(self.results_path, f"{epoch}_{i}.jpg"), nrow=4)
                            self.log_file.write(f"Iteration {global_iteration}: VQ Loss: {vq_loss.item():.3f}, GAN Loss: {gan_loss.item():.3f}, Perceptual Loss: {perceptual_loss.item():.3f}, Reconstruction Loss: {rec_loss.item():.3f}\n")
                            self.log_file.flush()

                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                    )
                    pbar.update(1)
                    global_iteration += 1

                torch.save(self.vqgan.module.state_dict(), os.path.join(self.checkpoints_path, f"vqgan_epoch_{epoch}.pt"))  
                if global_iteration > self.threshold:
                    torch.save(self.discriminator.module.state_dict(), os.path.join(self.checkpoints_path, f"discriminator_epoch_{epoch}.pt"))

        self.log_file.close()

if __name__ == "__main__":
    main()                
    
