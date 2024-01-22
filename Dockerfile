FROM python:3.11-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install and setup poetry
RUN pip install -U pip 
RUN apt-get update 
RUN apt install -y curl 
RUN apt install -y netcat-traditional
RUN apt install -y pipx
RUN apt install -y openssh-client
RUN pipx install poetry
ENV PATH="${PATH}:/root/.poetry/bin:/root/.local/bin"

WORKDIR /app
COPY src/. /app
COPY poetry.lock pyproject.toml /app/

WORKDIR /app
COPY . .
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

#CMD ["python3", "yolov7app-SAHI"]
