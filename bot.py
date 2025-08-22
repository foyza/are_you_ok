import logging
import os
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv("BOT_TOKEN")

if not API_TOKEN:
    raise ValueError("Не найден BOT_TOKEN")

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

YANDEX_API_URL = "https://speller.yandex.net/services/spellservice.json/checkText"

async def correct_text(text: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(YANDEX_API_URL, params={"text": text, "lang": "ru"}) as response:
            result = await response.json()
            if not result:
                return text
            corrected_text = text
            for item in result:
                if "s" in item and item["s"]:
                    corrected_text = corrected_text.replace(item["word"], item["s"][0])
            return corrected_text

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Отправь мне текст на русском, я проверю и исправлю ошибки.")

@dp.message_handler(content_types=types.ContentType.TEXT)
async def handle_text(message: types.Message):
    text = message.text.strip()
    corrected = await correct_text(text)
    if corrected != text:
        await message.reply(f"Исправленный текст:\n{corrected}")
    else:
        await message.reply("Ошибок не найдено!")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
