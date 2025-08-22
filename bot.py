import asyncio
from aiogram import Bot, Dispatcher, types
import joblib
import os

TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=TOKEN)
dp = Dispatcher()

model = joblib.load('model.joblib')

@dp.message()
async def classify_message(message: types.Message):
    text = message.text
    prediction = model.predict([text])[0]
    result = "😊 Вы счастливы!" if prediction == 1 else "😔 Вы несчастливы..."
    await message.answer(result)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
