FROM python:3.11-slim-bullseye

# Evita prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Instala wget para bajar pesos & build essentials mínimas
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    wget build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copia requisitos y los instala primero (mejor capa de cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de la fuente
COPY app ./app
COPY models ./backend/models

# Variables de entorno (puedes sobreescribirlas en docker-compose)
ENV OPENAI_API_KEY=""
ENV PYTHONUNBUFFERED=1

# Expone el puerto FastAPI
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

