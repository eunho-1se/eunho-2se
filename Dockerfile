FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install --default-timeout=2000 -r requirements.txt

COPY . .

EXPOSE 7777

ENTRYPOINT ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7777"]