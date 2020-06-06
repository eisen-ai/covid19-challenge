import shutil
import os
import tempfile
import boto3

from pydispatch import dispatcher
from eisen import EISEN_END_EPOCH_EVENT


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
        zip_filename = os.path.join(self.tempdir, 'artifacts.zip')

        shutil.make_archive(zip_filename, 'zip', self.artifacts_dir)

        without_prefix = self.s3_save_object[5:]

        broken_down = without_prefix.split('/')

        bucket = broken_down[0]

        object = '/'.join(broken_down[1:])

        for i in range(100):
            try:
                self.s3_client.upload_file(zip_filename, bucket, object)
            except:
                print('there was a problem uploading results to s3. Attempt {}'.format(i))
