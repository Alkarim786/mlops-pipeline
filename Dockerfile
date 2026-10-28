FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MLOPS_ROOT=/data MLOPS_MODEL=demo
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mlops ./mlops
VOLUME /data
EXPOSE 8000
CMD ["sh", "-c", "python -m mlops.bootstrap --root $MLOPS_ROOT --name $MLOPS_MODEL && uvicorn --factory mlops.app:_default_app --host 0.0.0.0 --port 8000"]
