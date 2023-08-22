from torch import nn


class SimpleDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = None,
        lin1_dim: int = 64,
        depth: int = 2,
    ):
        
        """
        Creates a decoder with varying depth and width.
        latent_dim -> lin1_dim -> lin1_dim*r -> lin1_dim*r^2 -> ... -> n_features(output_dim) 
        """

        super().__init__()

        self.latent_dim = latent_dim
        self.lin1_dim = lin1_dim
        self.output_dim = None
        self.depth = depth

        self.model = None

    def compute_model(self):
        r = pow(self.output_dim/self.lin1_dim, 1/self.depth)
        
        modules = []

        # First layer
        modules.append(nn.Linear(self.latent_dim, self.lin1_dim))
        modules.append(nn.LeakyReLU())

        # Middle layers
        for i in range(self.depth-1):
            modules.append(nn.Linear(round(self.lin1_dim*pow(r, i)), round(self.lin1_dim*pow(r, i+1))))
            modules.append(nn.LeakyReLU())

        # Last layer
        modules.append(nn.Linear(round(self.lin1_dim*pow(r, self.depth-1)), round(self.lin1_dim*pow(r, self.depth))))

        self.model = nn.Sequential(*modules)
        

    def set_dims(self, output_dim):
        self.output_dim = output_dim
        self.compute_model()

if __name__ == "__main__":
    _ = SimpleDecoder()
