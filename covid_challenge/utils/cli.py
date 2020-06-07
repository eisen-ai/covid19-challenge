import sys
import boto3
import os
import tempfile
import json
import signal

from time import sleep
from subprocess import Popen


def create_tb_termination_flag():
    with open('/tmp/results', 'w') as f:
        f.write('')


signal.signal(signal.SIGINT, create_tb_termination_flag)
signal.signal(signal.SIGTERM, create_tb_termination_flag)

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

while True:
    retcode = training.poll()

    sleep(10)

    if retcode is not None:
        create_tb_termination_flag()
        exit(retcode)
