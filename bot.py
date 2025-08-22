import joblib
from aiogram import Bot, Dispatcher, types
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

model = joblib.load("grammar_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@dp.message()
async def check_grammar(message: types.Message):
    text = message.text
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    conf = model.predict_proba(X)[0][pred]

    if pred == 1:
        reply = f"✅ Похоже, предложение грамматичное.\nConfidence: {conf:.2f}"
    else:
        reply = f"❌ Предложение выглядит неграмматично.\nConfidence: {conf:.2f}"
    await message.answer(reply)

if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot))
