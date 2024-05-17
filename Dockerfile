FROM python:3.12.3

WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "main.py"]