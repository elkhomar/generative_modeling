from torch import nn


class SimpleEncoder(nn.Module):
    def __init__(
        self,
        lin1_dim: int = 256,
        output_dim: int = 64,
        depth = 2
    ):
        """
        Creates and encoder with varying depth and width.
        n_features(input_dim) -> lin1_dim -> lin1_dim/r -> lin1_dim/r^2 -> ... -> output_dim
        """

        super().__init__()

        self.input_dim = None
        self.lin1_dim = lin1_dim
        self.output_dim = output_dim
        self.depth = depth

        self.model = None

    def compute_model(self):

        r = pow(self.output_dim/self.lin1_dim, 1/self.depth)
        modules = []

        # First layer
        modules.append(nn.Linear(self.input_dim, self.lin1_dim))
        modules.append(nn.LeakyReLU())

        # Middle layers
        for i in range(self.depth):
            modules.append(nn.Linear(round(self.lin1_dim*pow(r, i)), round(self.lin1_dim*pow(r, i+1))))
            modules.append(nn.LeakyReLU())

        self.model = nn.Sequential(*modules)

    def set_dims(self, input_dim):
        self.input_dim = input_dim
        self.compute_model()

if __name__ == "__main__":
    _ = SimpleEncoder()
