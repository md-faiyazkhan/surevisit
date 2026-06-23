FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

ENV PIP_DEFAULT_TIMEOUT=300
RUN pip install --no-cache-dir --retries 10 -r requirements.txt

COPY app/ app/
COPY dashboard/ dashboard/
COPY models/ models/

EXPOSE 8000
EXPOSE 8501

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run dashboard/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]