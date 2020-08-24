FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN apt-get install -y git

RUN pip install --upgrade git+https://github.com/eisen-ai/eisen-deploy.git --no-dependencies
RUN pip install --upgrade git+https://github.com/eisen-ai/eisen-cli.git --no-dependencies
RUN pip install --upgrade git+https://github.com/eisen-ai/eisen-extras.git --no-dependencies
RUN pip install --upgrade git+https://github.com/eisen-ai/eisen-core.git
RUN pip install Click six Pillow msgpack dill

RUN pip install git+https://github.com/eisen-ai/covid19-challenge.git