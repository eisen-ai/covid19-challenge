import boto3
import os
import nibabel as nib
import logging

from covid_challenge import get_file_from_s3


class LoadS3Nifti:
    """
    This transform loads Nifti data from the cloud and specifically from S3.

    .. code-block:: python
        from eisen.io import LoadS3Nifti
        tform = LoadS3Nifti(['image', 'label'], 's3://bucket/path/')
    """

    def __init__(self, fields, data_dir, canonical=False, aws_id=None, aws_secret=None, cache='/cache'):
        self.cache = cache

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_id,
            aws_secret_access_key=aws_secret
        )

        self.fields = fields
        self.data_dir = data_dir
        self.canonical = canonical

        assert self.data_dir[0:5] == 's3://'

    def __call__(self, data):
        """
        :param data: Data dictionary to be processed by this transform
        :type data: dict
        :return: Updated data dictionary
        :rtype: dict
        """
        for field in self.fields:

            complete_file_path = os.path.join(self.data_dir, data[field])

            assert complete_file_path[0:5] == 's3://'

            filename = get_file_from_s3(self.s3_client, complete_file_path, self.cache)

            img = nib.load(filename)

            logging.info(filename)
            print(filename)

            if self.canonical:
                img = nib.as_closest_canonical(img)

            data[field] = img
            data[field + '_affines'] = img.affine
            data[field + '_orientations'] = nib.aff2axcodes(img.affine)

        return data