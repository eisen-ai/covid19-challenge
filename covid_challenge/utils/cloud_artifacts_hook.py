import shutil
import os
import tempfile
import boto3

from pydispatch import dispatcher
from eisen import EISEN_END_EPOCH_EVENT
from covid_challenge import put_file_on_s3


class SaveCloudArtifactsHook:
    def __init__(self, workflow_id, phase, artifacts_dir, s3_save_object, aws_id=None, aws_secret=None,):
        dispatcher.connect(self.end_epoch, signal=EISEN_END_EPOCH_EVENT, sender=workflow_id)

        self.phase = phase
        self.workflow_id = workflow_id
        self.artifacts_dir = artifacts_dir
        self.s3_save_object = s3_save_object
        self.tempdir = tempfile.mkdtemp()

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_id,
            aws_secret_access_key=aws_secret
        )

    def __del__(self):
        self.end_epoch(None)

        shutil.rmtree(self.tempdir)

    def end_epoch(self, message):
        filename = os.path.join(self.tempdir, 'artifacts')

        zip_filename = shutil.make_archive(filename, 'zip', self.artifacts_dir + '/*')

        put_file_on_s3(self.s3_client, zip_filename, self.s3_save_object)