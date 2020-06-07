from flask import Flask
from subprocess import Popen
from multiprocessing import Process


tb_cmd = [
    'tensorboard',
    '--logdir=/tmp/results',
    '--host=0.0.0.0',
    '--port=80'
]

print('I am about to run Tensorboard via: {}'.format(tb_cmd))

tb_process = Popen(tb_cmd)

app = Flask(__name__)

server = Process(target=app.run, kwargs={'host': '0.0.0.0', 'port': '6006'})

print(server)

@app.route('/')
def hello_world():
    tb_process.terminate()

    return 'Terminated'

server.start()


while True:
    retcode = tb_process.poll()

    if retcode is not None:
        server.terminate()
        exit(0)
