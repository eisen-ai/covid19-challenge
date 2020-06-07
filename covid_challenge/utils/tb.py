from flask import Flask

from subprocess import Popen


tb_cmd = [
    'tensorboard',
    '--logdir=/tmp/results',
    '--host=0.0.0.0',
    '--port=80'
]

print('I am about to run Tensorboard via: {}'.format(tb_cmd))

tb_process = Popen(tb_cmd)

app = Flask(__name__)

@app.route('/')
def hello_world():
    tb_process.terminate()
    exit(0)

app.run(host='0.0.0.0', port='6006')
