import os
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from dotenv import load_dotenv
import joblib

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

bot = Bot(token=TOKEN)
dp = Dispatcher()

model = joblib.load("model.pkl")

kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Классифицировать")],
    ],
    resize_keyboard=True
)

@dp.message(commands=["start"])
async def start(message: types.Message):
    await message.answer("Привет! Отправь текст для классификации:", reply_markup=kb)

@dp.message()
async def classify_text(message: types.Message):
    text = message.text
    # Здесь предполагаем, что модель принимает список текстов
    prediction = model.predict([text])[0]
    await message.answer(f"Результат классификации: {prediction}")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
