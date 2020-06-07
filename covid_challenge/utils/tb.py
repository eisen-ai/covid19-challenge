import os

from subprocess import Popen


tb_cmd = [
    'tensorboard',
    '--logdir /tmp/results',
    '--host 0.0.0.0',
    '--port 80'
]

print('I am about to run Tensorboard via: {}'.format(tb_cmd))

tb_process = Popen(tb_cmd)

while True:
    retcode = tb_process.poll()

    if os.path.exists('/tmp/results/terminate'):
        tb_process.terminate()

        exit(tb_process.poll())
