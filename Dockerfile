# Use official Python image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy the project files
COPY pyproject.toml poetry.lock ./

# Install dependencies using poetry
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi

# Copy the rest of the files
COPY . .

# Set right work directory
WORKDIR /app/src

# Expose ports - FastAPI 7860 Streamlit 8501
EXPOSE 7860 8501

# RUN FastAPI and Streamlit in parallel
CMD uvicorn app:app --host 0.0.0.0 port 7860 && streamlit run app.py --server.port 8501 server.enableCORS false
