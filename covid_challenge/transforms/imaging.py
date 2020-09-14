import copy

import numpy as np
from torchvision.transforms import Compose

from eisen.transforms import FilterFields, StackImagesChannelwise, CreateConstantFlags


class ModCreateConstantFlags:
    """Transform to create new fields or modify existing keys in the data dictionary containing.
    """

    def __init__(self, fields, values, force=True):
        """
        :param fields: names of the fields of data dictionary to work on
        :type fields: list of str
        :param values: list of values to add to data
        :type values: list of values
        :param force: If True, this key and value will always be added to dict, else will only add this key and value
            if key initially exists in dict.
        :type force: bool
        """
        self.fields = fields
        self.values = values
        self.force = force

        assert len(fields) == len(values)

    def __call__(self, data):
        """
        :param data: Data dictionary to be processed by this transform
        :type data: dict
        :return: Updated data dictionary
        :rtype: dict
        """
        for field, value in zip(self.fields, self.values):
            if self.force or field in data.keys():
                data[field] = value

        return data


class KeyFriendlyCompose(Compose):
    r"""Compose transform for input dict with incomplete set of fieldnames.

    This compose transform will allow incomplete set of keynames to exist within the input dict. For example, training
    of labelled and unlabelled inputs.
    """
    def __init__(self, transforms_list):
        r"""
        :param transforms_list: List of transforms
        """
        super(KeyFriendlyCompose, self).__init__(transforms_list)

    def __call__(self, inp):

        for t in self.transforms:
            cur_keys = copy.deepcopy(t.fields)

            if isinstance(t, FilterFields):
                inp = {k: v for k, v in inp.items() if k in cur_keys}

            elif isinstance(t, StackImagesChannelwise):
                composite_image = []
                for key in cur_keys:
                    if key in inp.keys():
                        composite_image.append(inp[key])

                if len(composite_image) != 0:
                    if t.create_new_dim:
                        inp[t.dst_field] = np.stack(composite_image, axis=0)
                    else:
                        inp[t.dst_field] = np.concatenate(composite_image, axis=0)

            else:
                for key in cur_keys:
                    t.fields = [key]
                    if key in inp.keys() or isinstance(t, (CreateConstantFlags, ModCreateConstantFlags)):
                        inp.update(t(inp))
            t.fields = cur_keys

        return inp
