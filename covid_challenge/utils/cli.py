import sys
import boto3
import os
import tempfile
import json
import signal
import requests

from time import sleep
from subprocess import Popen

from covid_challenge.utils.secret_ops import get_secret


backend_key = get_secret('JOB_STATUS_UPDATE_KEY')['key']


def terminate_tensorboard():
    requests.get('http://127.0.0.1:6006/')


signal.signal(signal.SIGINT, terminate_tensorboard)
signal.signal(signal.SIGTERM, terminate_tensorboard)

s3 = boto3.client('s3')

config = sys.argv[1]
config_epoch = sys.argv[2]
data = sys.argv[3]
results = sys.argv[4]

config_path = os.path.join(results, 'config.json')

os.makedirs(results)

config = json.loads(config)

dirpath = tempfile.mkdtemp()

config_path = os.path.join(dirpath, 'config.json')

with open(config_path, 'w') as f:
    json.dump(config, f)


eisen_cmd = ['python3',
             '/opt/conda/bin/eisen',
             'train',
             config_path,
             str(config_epoch),
             '--data_dir={}'.format(data),
             '--artifact_dir={}'.format(results)
             ]

print('I am about to run Eisen via: {}'.format(eisen_cmd))

training = Popen(eisen_cmd)

job_id = os.listdir('/tmp/results')[0]

r = requests.post(
    'https://e2chj08pf8.execute-api.eu-central-1.amazonaws.com/v0/update-job-status',
    json={'job_id': job_id, 'status': 'Running', 'key': backend_key}
)

while True:
    retcode = training.poll()

    sleep(10)

    if retcode is not None:
        terminate_tensorboard()

        if retcode == 0:
            r = requests.post(
                'https://e2chj08pf8.execute-api.eu-central-1.amazonaws.com/v0/update-job-status',
                json={'job_id': job_id, 'status': 'Success', 'key': backend_key}
            )
        else:
            r = requests.post(
                'https://e2chj08pf8.execute-api.eu-central-1.amazonaws.com/v0/update-job-status',
                json={'job_id': job_id, 'status': 'Failed', 'key': backend_key}
            )

        exit(retcode)
