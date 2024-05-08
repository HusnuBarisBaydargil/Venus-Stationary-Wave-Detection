import torch
import torch.nn as nn

class Codebook(nn.Module):
    def __init__(self):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = 256
        self.latent_dim = 128 * 8 * 8
        self.beta = 0.25

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss
    
# # Create a dummy input tensor
# dummy_input = torch.randn(1, 128, 8, 8)
# # Initialize the codebook
# codebook = Codebook()

# # Forward pass of the dummy input through the codebook
# quantized_output, min_encoding_indices, vq_loss = codebook(dummy_input)

# # Print the outputs
# print("Quantized output shape:", quantized_output.shape)
# print("Minimum encoding indices shape:", min_encoding_indices.shape)
# print("Vector quantization loss:", vq_loss)