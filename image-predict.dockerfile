FROM python:3.10.12-slim

RUN pip install -U pip

WORKDIR /app

RUN pip install flask==3.0.0 gunicorn==21.2.0
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

COPY ["DINOClassifier.py", "predict.py", "model_v1.pth", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]