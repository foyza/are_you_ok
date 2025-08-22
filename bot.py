import os, asyncio, joblib
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise RuntimeError("Set BOT_TOKEN in .env")

bot = Bot(token=TOKEN)
dp = Dispatcher()
model = joblib.load("model.joblib")

@dp.message(Command("start"))
async def start_cmd(m: Message):
    await m.answer("Привет! Отправь английское предложение, и я скажу, грамматично ли оно.")

@dp.message()
async def check_grammar(m: Message):
    text = m.text.strip()
    proba = model.predict_proba([text])[0][1]
    label = "✅ грамматично" if proba >= 0.5 else "❌ неграмматично"
    await m.answer(f"{label}\nConfidence: {proba:.2f}")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
