FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.enableCORS=false"]