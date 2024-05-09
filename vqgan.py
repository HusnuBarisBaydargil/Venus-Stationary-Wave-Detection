import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook

class VQGAN(nn.Module):
    def __init__(self, device):
        super(VQGAN, self).__init__()
        self.device = device
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.codebook = Codebook().to(self.device)
        self.quant_conv = nn.Conv2d(128, 128, 1).to(self.device)
        self.post_quant_conv = nn.Conv2d(128, 128, 1).to(self.device)
        
    def forward(self, imgs):
        encoded_images, skips = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping, skips[::-1])  
        
        return decoded_images, codebook_indices, q_loss

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_conv_layer = self.decoder.final_conv
        last_layer_weight = last_conv_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))