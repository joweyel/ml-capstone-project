FROM python:3.10-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ./streamlit_app.py /app

CMD ["streamlit", "run", "streamlit_app.py"]