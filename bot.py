import logging
import os
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Загружаем модель
model = load_model("model.h5")
tokenizer = Tokenizer(num_words=1000)

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Привет! Напиши текст, а я оценю, счастлив ты или нет 🙂")

@dp.message()
async def analyze(message: types.Message):
    text = message.text
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=10)
    pred = model.predict(seq)[0][0]
    if pred > 0.5:
        await message.answer(f"😊 Ты выглядишь счастливым! ({pred:.2f})")
    else:
        await message.answer(f"😔 Ты выглядишь грустным. ({pred:.2f})")

if __name__ == "__main__":
    import asyncio
    asyncio.run(dp.start_polling(bot))
