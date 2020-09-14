import torch
from eisen.ops.losses import DiceLoss


class DiceMSELoss(torch.nn.Module):
    def __init__(self, seg_wt=1.0, dim=None):
        super(DiceMSELoss, self).__init__()
        self.sum_kwargs = {}
        if dim is not None:
            self.sum_kwargs['dim'] = dim
        self.seg_wt = seg_wt
        self.mse_loss = torch.nn.MSELoss()
        self.dice_loss = DiceLoss(weight=seg_wt)
        self.eps = torch.finfo(type=torch.float32).eps

    def forward(self, image, prediction, reconstruction, seg_target):

        mse_loss = self.mse_loss(image, reconstruction)

        # Calculate segmentation loss
        seg_mask = seg_target.sum([1, 2, 3, 4]) != 0.0
        if seg_mask.sum() != 0:
            prediction = prediction[seg_mask]
            seg_target = seg_target[seg_mask]
            dice_loss = self.dice_loss(prediction, seg_target)
        else:
            dice_loss = torch.tensor(0.0).to(mse_loss.device)

        total_loss = dice_loss + (1 - self.seg_wt) * mse_loss
        return total_loss
