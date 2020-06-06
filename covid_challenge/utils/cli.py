import sys
import boto3
import os
import tempfile

from time import sleep
from subprocess import Popen


s3 = boto3.client('s3')

config = sys.argv[1]
config_epoch = sys.argv[2]
data = sys.argv[3]
results = sys.argv[4]

config_path = os.path.join(results, 'config.json')

os.makedirs(results)

with open(config_path, 'wb') as f:
    s3.download_fileobj('configurations-challenge', config, f)

eisen_cmd = ['python3',
             '/opt/conda/bin/eisen',
             'train',
             str(config_path),
             str(config_epoch),
             '--data_dir={}'.format(data),
             '--artifact_dir={}'.format(results)
             ]

print('I am about to run Eisen via: {}'.format(eisen_cmd))

training = Popen(eisen_cmd)

dirpath = tempfile.mkdtemp()


while True:
    retcode = training.poll()

    sleep(300)

    if retcode is not None:
        exit(retcode)
