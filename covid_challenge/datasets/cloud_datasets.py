import boto3
import os
import tempfile

from eisen.datasets import MSDDataset, JsonDataset
from eisen.utils import read_json_from_file
from covid_challenge import get_file_from_s3


class S3MSDDataset(MSDDataset):
    def __init__(self, data_dir, json_file, phase, aws_id=None, aws_secret=None, transform=None):

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_id,
            aws_secret_access_key=aws_secret
        )

        self.tempdir = tempfile.mkdtemp()

        json_file = get_file_from_s3(self.s3_client, os.path.join(data_dir, json_file), self.tempdir)

        msd_dataset = read_json_from_file(json_file)

        self.json_dataset = msd_dataset[phase]

        msd_dataset.pop('training', None)
        msd_dataset.pop('test', None)

        if phase == 'test':
            # test images are stored as list of filenames instead of dictionaries. Need to convert that.
            dset = []
            for elem in self.json_dataset:
                dset.append({'image': elem})

            self.json_dataset = dset

        self.attributes = msd_dataset

        self.transform = transform


class S3JsonDataset(JsonDataset):

    def __init__(self, data_dir, json_file, aws_id=None, aws_secret=None, transform=None):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_id,
            aws_secret_access_key=aws_secret
        )

        self.tempdir = tempfile.mkdtemp()

        json_file = get_file_from_s3(self.s3_client, os.path.join(data_dir, json_file), self.tempdir)

        self.json_dataset = read_json_from_file(json_file)

        self.transform = transform

