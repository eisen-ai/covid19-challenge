from torch import nn
import torch

from eisen.models import GroupNorm3D, conv_block_2_3d, max_pooling_3d, conv_trans_block_3d, conv_block_3d


class ModUNet3D(nn.Module):
    """Modified UNet3D which outputs segmentation masks and reconstructed volume.

    An example modified UNet3d which could be used for self-supervised learning when no ground-truth label mask is
    available.

    """

    def __init__(
            self,
            input_channels,
            output_channels,
            n_filters=16,
            seg_outputs_activation='sigmoid',
            rec_output_activation='none',
            normalization='groupnorm'
    ):
        """
        :param input_channels: number of input channels
        :type input_channels: int
        :param output_channels: number of output channels
        :type output_channels: int
        :param n_filters: number of filters
        :type n_filters: int
        :param seg_outputs_activation: segmentation output activation type either sigmoid, softmax or none
        :type seg_outputs_activation: str
        :param rec_output_activation: reconstruction activation type either sigmoid, softmax or none
        :type rec_output_activation: str
        :param normalization: normalization either groupnorm, batchnorm or none
        :type normalization: str
        """
        super(ModUNet3D, self).__init__()

        self.in_dim = input_channels
        self.out_dim = output_channels
        self.num_filters = n_filters

        if normalization == 'groupnorm':
            normalization = GroupNorm3D
        elif normalization == 'batchnorm':
            normalization = nn.BatchNorm3d
        else:
            normalization = nn.Identity

        activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation, normalization)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation, normalization)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation, normalization)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation, normalization)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation, normalization)
        self.pool_5 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation, normalization)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation, normalization)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation, normalization)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation, normalization)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation, normalization)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation, normalization)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation, normalization)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation, normalization)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation, normalization)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation, normalization)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation, normalization)

        # Output
        self.out_reconstruction = conv_block_3d(self.num_filters, self.in_dim, activation, nn.Identity)
        self.out_segmentation = conv_block_3d(self.num_filters, self.out_dim, activation, nn.Identity)

        if seg_outputs_activation == 'sigmoid':
            self.outputs_activation_fn = nn.Sigmoid()
        elif seg_outputs_activation == 'softmax':
            self.outputs_activation_fn = nn.Softmax()
        else:
            self.outputs_activation_fn = nn.Identity()

        if rec_output_activation == 'sigmoid':
            self.outputs_activation_fn_rec = nn.Sigmoid()
        elif rec_output_activation == 'softmax':
            self.outputs_activation_fn_rec = nn.Softmax()
        else:
            self.outputs_activation_fn_rec = nn.Identity()

    def forward(self, x):
        """
        Computes output of the network.

        :param x: Input tensor containing images
        :type x: torch.Tensor
        :return: segmentation_output_tensor, reconstruction_output_tensor
        """
        # Down sampling
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        down_5 = self.down_5(pool_4)
        pool_5 = self.pool_5(down_5)

        # Bridge
        bridge = self.bridge(pool_5)

        # Up sampling
        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_5], dim=1)
        up_1 = self.up_1(concat_1)

        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_4], dim=1)
        up_2 = self.up_2(concat_2)

        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_3], dim=1)
        up_3 = self.up_3(concat_3)

        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4, down_2], dim=1)
        up_4 = self.up_4(concat_4)

        trans_5 = self.trans_5(up_4)
        concat_5 = torch.cat([trans_5, down_1], dim=1)
        up_5 = self.up_5(concat_5)

        # Output
        out_1 = self.out_segmentation(up_5)
        out_2 = self.out_reconstruction(up_5)

        return self.outputs_activation_fn(out_1), self.outputs_activation_fn_rec(out_2)
