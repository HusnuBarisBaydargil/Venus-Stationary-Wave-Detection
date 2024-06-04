import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import load_data, weights_init
from datetime import datetime

class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        self.prepare_folders()

        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_vq, opt_disc

    def prepare_folders(self):
        date_today = datetime.now().strftime("%Y-%m-%d_%H%M")
        base_dir = os.path.join(os.getcwd(), "Experiments", f"{date_today}")
        os.makedirs(base_dir, exist_ok=True)
        print(f"Experiment base directory created at: {base_dir}")

        self.results_path = os.path.join(base_dir, "results")
        self.checkpoints_path = os.path.join(base_dir, "checkpoints")
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.checkpoints_path, exist_ok=True)

        self.log_file_path = os.path.join(base_dir, "training_log.txt")
        self.log_file = open(self.log_file_path, "a")
        print(f"Logging to {self.log_file_path}")

    def train(self, args):
        train_dataset = load_data(args.dataset_path, args.batch_size, image_size=args.image_size)
        steps_per_epoch = len(train_dataset)
        global_iteration = 0 

        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    imgs = imgs.to(device=args.device)
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch*steps_per_epoch+i, threshold=args.disc_start)

                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    if i % 100 == 0:
                        with torch.no_grad():
                            real_images = (imgs[:4] + 1) / 2
                            fake_images = (decoded_images[:4] + 1) / 2
                            real_fake_images = torch.cat((real_images, fake_images), dim=0)
                            vutils.save_image(real_fake_images, os.path.join(self.results_path, f"{epoch}_{i}.jpg"), nrow=4)

                            vq_loss_value = vq_loss.mean().item()
                            gan_loss_value = gan_loss.mean().item()
                            perceptual_loss_value = perceptual_loss.mean().item() 
                            rec_loss_value = rec_loss.mean().item()

                            self.log_file.write(
                                f"Iteration {global_iteration}: VQ Loss: {vq_loss_value:.5f}, "
                                f"GAN Loss: {gan_loss_value:.5f}, Perceptual Loss: {perceptual_loss_value:.5f}, "
                                f"Reconstruction Loss: {rec_loss_value:.5f}\n"
                            )
                            self.log_file.flush()

                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                    )
                    pbar.update(1)
                    global_iteration += 1

                torch.save(self.vqgan.state_dict(), os.path.join(self.checkpoints_path, f"vqgan_epoch_{epoch}.pt"))
                if global_iteration > args.disc_start:
                    torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoints_path, f"discriminator_epoch_{epoch}.pt"))

        self.log_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=128, help='Image height and width (default: 128)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 1024)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 1)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda:2", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=6, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 10000)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    args = parser.parse_args()
    args.dataset_path = "path/to/dataset"

    train_vqgan = TrainVQGAN(args)
