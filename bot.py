import os
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from dotenv import load_dotenv
import joblib
import numpy as np

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

bot = Bot(token=TOKEN)
dp = Dispatcher()

model = joblib.load("model.pkl")

kb = ReplyKeyboardMarkup(resize_keyboard=True)
kb.add(KeyboardButton("Классифицировать"))

@dp.message_handler(commands=["start"])
async def start(message: types.Message):
    await message.answer("Привет! Отправь текст, и я предскажу его класс.", reply_markup=kb)

@dp.message_handler(lambda message: message.text != "Классифицировать")
async def classify(message: types.Message):
    text = message.text
    # Модель ждет вектор, здесь просто пример с длиной текста
    X = np.array([[len(text)]])
    pred = model.predict(X)[0]
    await message.answer(f"Класс текста: {pred}")

@dp.message_handler(lambda message: message.text == "Классифицировать")
async def ask_input(message: types.Message):
    await message.answer("Отправь текст для классификации.")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(dp.start_polling(bot))
