import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

# Загружаем модель и токенайзер
model = tf.keras.models.load_model("model.h5")
with open("tokenizer.pkl", "rb") as f:
    word_index = pickle.load(f)

max_length = 200

def text_to_sequence(text):
    tokens = [word_index.get(word, 0) for word in text.lower().split()]
    return pad_sequences([tokens], maxlen=max_length)

@dp.message_handler(commands=["start"])
async def start(message: types.Message):
    await message.answer("Привет! Отправь текст, а я скажу, насколько он счастливый 😊")

@dp.message_handler()
async def predict(message: types.Message):
    seq = text_to_sequence(message.text)
    prediction = model.predict(seq)[0][0]
    if prediction > 0.6:
        mood = "Счастливый 😍"
    elif prediction > 0.4:
        mood = "Нейтральный 🙂"
    else:
        mood = "Грустный 😢"
    await message.answer(f"Твой текст выглядит как: {mood} (уверенность {prediction:.2f})")

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
