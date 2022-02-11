FROM pytorch/pytorch:latest

WORKDIR /workspace
ADD requirements.txt .
ADD run_test.sh .
RUN pip3 install -r requirements.txt

ENV PYTHONPATH $PYTHONPATH:/workspace