# bot.py
import os
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import Command
from dotenv import load_dotenv
import joblib

# Загрузка переменных окружения
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Инициализация бота и диспетчера
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Загрузка обученной модели
model = joblib.load("model.pkl")

# Создание клавиатуры
kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Классифицировать")],
    ],
    resize_keyboard=True
)

# Обработчик команды /start
@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Привет! Отправь текст для классификации:", reply_markup=kb)

# Обработчик текстовых сообщений
@dp.message()
async def classify_text(message: types.Message):
    text = message.text
    # Предполагаем, что модель принимает список текстов
    prediction = model.predict([text])[0]
    await message.answer(f"Результат классификации: {prediction}")

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
