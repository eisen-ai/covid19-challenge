import numpy as np
import torch
import math

from eisen.utils.logging.summaries import TensorboardSummaryHook as TBSH


class TensorboardSummaryHook(TBSH):

    def write_volumetric_image(self, name, value, global_step):
        self.writer.add_scalar(name + "/mean", np.mean(value), global_step=global_step)
        self.writer.add_scalar(name + "/std", np.std(value), global_step=global_step)
        self.writer.add_histogram(name + "/histogram", value.flatten(), global_step=global_step)

        # value is currently [N, Ch, W, H, D]. apply watermark

        watermark_size = 15

        p = 0
        for i in range(int(math.floor(value.shape[2] / watermark_size))):
            for j in range(int(math.floor(value.shape[3] / watermark_size))):
                for z in range(int(math.floor(value.shape[4] / watermark_size))):
                    if p % 2 == 0:
                        value[
                                i * watermark_size:(i + 1) * watermark_size,
                                j * watermark_size:(j + 1) * watermark_size,
                                z * watermark_size:(z + 1) * watermark_size,
                            ] = 0

                    p = p + 1

        v = np.transpose(value, [0, 2, 1, 3, 4])

        if v.shape[2] != 3 and v.shape[2] != 1:
            v = np.average(v, axis=2, weights=np.arange(0, 1, 1 / v.shape[2]))[:, :, np.newaxis]

        torch_value = torch.tensor(v).float()

        self.writer.add_video(name + "_axis_1", torch_value, fps=10, global_step=global_step)

        if self.show_all_axes:
            v = np.transpose(value, [0, 3, 1, 2, 4])

            if v.shape[2] != 3 and v.shape[2] != 1:
                v = np.average(v, axis=2, weights=np.arange(0, 1, 1 / v.shape[2]))[:, :, np.newaxis]

            torch_value = torch.tensor(v).float()

            self.writer.add_video(name + "_axis_2", torch_value, fps=10, global_step=global_step)

            v = np.transpose(value, [0, 4, 1, 2, 3])

            if v.shape[2] != 3 and v.shape[2] != 1:
                v = np.average(v, axis=2, weights=np.arange(0, 1, 1 / v.shape[2]))[:, :, np.newaxis]

            torch_value = torch.tensor(v).float()

            self.writer.add_video(name + "_axis_3", torch_value, fps=10, global_step=global_step)
