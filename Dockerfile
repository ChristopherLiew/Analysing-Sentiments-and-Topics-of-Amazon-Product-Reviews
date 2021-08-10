FROM python:3.8

RUN pip install poetry
RUN mkdir /amz-sent-analysis

WORKDIR /amz-sent-analysis
ENV PYTHONPATH=${PYTHONPATH}:${PWD}

COPY poetry.lock pyproject.toml ./
RUN poetry export -f requirements.txt --output requirements.txt
RUN pip install -r requirements.txt
COPY . ./

RUN poetry build
RUN python -m pip install --upgrade pip
RUN pip install dist/src-*.whl
# Copies app dependencies into container wd
# Do not create a venv first since we are already in a venv by using a docker image

# We use pip and do not use poetry config venv create false as it will create an editable install
# RUN poetry config virtualenvs.create false
# RUN poetry install --no-ansi --no-interaction
