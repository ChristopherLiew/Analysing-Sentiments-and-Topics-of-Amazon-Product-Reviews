FROM python:3.8

RUN mkdir /analysing-sentiments-and-topics-of-amazon-reviews

COPY /analysing-sentiments-and-topics-of-amazon-reviews /analysing-sentiments-and-topics-of-amazon-reviews
# Copies app dependencies into container wd
COPY pyproject.toml /analysing-sentiments-and-topics-of-amazon-reviews

WORKDIR /analysing-sentiments-and-topics-of-amazon-reviews
ENV PYTHONPATH=${PYTHONPATH}:${PWD}

RUN pip3 install poetry
# Do not create a venv first since we are already in a venv by using a docker image
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev
