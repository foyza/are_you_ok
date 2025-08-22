import logging
import os
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv("BOT_TOKEN")

if not API_TOKEN:
    raise ValueError("Не найден BOT_TOKEN в .env")

# Логирование
logging.basicConfig(level=logging.INFO)

# Инициализация бота
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# Загружаем модель для исправления русского текста
MODEL_NAME = "cointegrated/rut5-base-spell-correct"
logging.info(f"Загружаем модель {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
logging.info("Модель загружена успешно.")

def correct_text(text: str) -> str:
    try:
        inputs = tokenizer.encode(text, return_tensors="pt", truncation=True)
        outputs = model.generate(inputs, max_length=512)
        corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected
    except Exception as e:
        logging.error(f"Ошибка при исправлении текста: {e}")
        return text

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply(
        "Привет! Отправь мне текст на русском, и я исправлю ошибки с помощью нейросети."
    )

@dp.message_handler(content_types=types.ContentType.TEXT)
async def handle_text(message: types.Message):
    text = message.text.strip()
    corrected = correct_text(text)
    if corrected != text:
        await message.reply(f"Исправленный текст:\n{corrected}")
    else:
        await message.reply("Ошибок не найдено!")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
