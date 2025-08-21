import os
import cv2
import numpy as np
import pandas as pd
from aiogram import Bot, Dispatcher, types
from aiogram.types import InputFile
from aiogram.filters import Command
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

MODE = "predict"  # change to "collect" for data collection
IMG_SIZE = 64
DATASET_DIR = "dataset/images"
LABELS_FILE = "dataset/labels.csv"

if MODE == "predict":
    model = load_model("model.h5")

os.makedirs(DATASET_DIR, exist_ok=True)
if not os.path.exists("dataset"):
    os.makedirs("dataset")

if MODE == "collect" and not os.path.exists(LABELS_FILE):
    pd.DataFrame(columns=["filename", "happiness"]).to_csv(LABELS_FILE, index=False)

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    if MODE == "collect":
        await message.reply("Отправьте фото селфи, затем укажите ваш уровень счастья (0-100).")
    else:
        await message.reply("Отправьте фото, я предскажу happiness score (0-100%).")

@dp.message(types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    photo = message.photo[-1]
    file_path = await photo.download()
    local_path = file_path.name

    if MODE == "collect":
        await message.reply("Введите ваш уровень счастья (0-100):")
        dp['last_photo'] = local_path
    else:
        img = cv2.imread(local_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        pred = model.predict(np.expand_dims(img, axis=0))[0][0]
        pred = int(max(0, min(100, pred)))
        await message.reply(f"Предсказанный happiness score: {pred}%")
        os.remove(local_path)

@dp.message(lambda m: m.text.isdigit() and MODE == "collect")
async def handle_label(message: types.Message):
    score = int(message.text)
    img_path = dp.get('last_photo')
    if img_path:
        new_name = f"{len(os.listdir(DATASET_DIR))+1}.jpg"
        os.rename(img_path, os.path.join(DATASET_DIR, new_name))
        df = pd.read_csv(LABELS_FILE)
        df = pd.concat([df, pd.DataFrame([[new_name, score]], columns=["filename","happiness"])])
        df.to_csv(LABELS_FILE, index=False)
        await message.reply("Фото и метка сохранены.")
    else:
        await message.reply("Сначала отправьте фото.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(dp.start_polling(bot))
