FROM --platform=arm64  jupyter/tensorflow-notebook:python-3.11
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt