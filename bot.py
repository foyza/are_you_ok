# bot.py
import os, asyncio, joblib
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise RuntimeError("Set BOT_TOKEN (or TELEGRAM_TOKEN) in environment or .env")

bot = Bot(token=TOKEN)
dp = Dispatcher()
model = joblib.load("model.joblib")

@dp.message(Command("start"))
async def on_start(m: Message):
    await m.answer("Напиши английское предложение — скажу, грамматично ли оно.\nExample: “The book on the table is mine.”")

@dp.message()
async def check_grammar(m: Message):
    text = (m.text or "").strip()
    if not text:
        await m.answer("Пришли текст сообщением.")
        return
    proba = getattr(model, "predict_proba", None)
    if proba:
        p = float(model.predict_proba([text])[0][1])
        label = "✅ грамматично" if p >= 0.5 else "❌ неграмматично"
        await m.answer(f"{label}\nConfidence: {p:.2f}")
    else:
        pred = int(model.predict([text])[0])
        label = "✅ грамматично" if pred == 1 else "❌ неграмматично"
        await m.answer(label)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
