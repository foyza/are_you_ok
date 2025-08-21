FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Если model.h5 не существует – тренируем
RUN python train_model.py || echo "Model already exists"

CMD ["python", "bot.py"]
