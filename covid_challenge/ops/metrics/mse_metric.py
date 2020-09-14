import torch


class MSEMetric(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(MSEMetric, self).__init__()
        self.sum_kwargs = {}
        self.weight = weight
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, image, reconstruction):
        mse_loss = self.mse_loss(image, reconstruction)
        total_loss = self.weight * mse_loss
        return total_loss
