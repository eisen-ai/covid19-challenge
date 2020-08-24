FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN apt-get install -y git

RUN pip install --upgrade git+https://github.com/eisen-ai/eisen-deploy.git
RUN pip install --upgrade git+https://github.com/eisen-ai/eisen-cli.git
RUN pip install --upgrade git+https://github.com/eisen-ai/eisen-extras.git
RUN pip install --upgrade git+https://github.com/eisen-ai/eisen-core.git

RUN pip install git+https://github.com/eisen-ai/covid19-challenge.git