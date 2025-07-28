FROM python:3.10-slim

WORKDIR /app

COPY src/ ./src/
COPY model/ ./model/
COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Ensure Python recognizes src as a package
RUN touch src/__init__.py

RUN mkdir -p /app/dataset/input /app/dataset/output

ENV MODEL_DIR=/app/model
ENV INPUT_DIR=/app/dataset/input
ENV OUTPUT_DIR=/app/dataset/output

CMD ["python", "src/evaluate_model.py", "/app/dataset/input", "/app/dataset/output"]
