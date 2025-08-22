import asyncio
from aiogram import Bot, Dispatcher, types
from deeppavlov import build_model, configs
import os
from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

model = build_model(configs.syntax.syntax_ru, download=True)

@dp.message()
async def check_grammar(message: types.Message):
    text = message.text
    corrected = model([text])[0]
    if text == corrected:
        await message.answer(f"✅ Предложение грамматично.\nИсправлений нет.")
    else:
        await message.answer(f"❌ Предложение содержит ошибки.\nИсправленная версия:\n{corrected}")

if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot, skip_updates=True))
