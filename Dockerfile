FROM python:3.11.4-slim-bullseye

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock /app/

RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi --only main

COPY . /app

EXPOSE 8080

CMD ["uvicorn", "ml_model_monitoring.model_deployment.api:app", "--host", "0.0.0.0", "--port", "8080"]
