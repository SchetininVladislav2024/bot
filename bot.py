import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Токен из переменной окружения
TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=TOKEN)
dp = Dispatcher()

# Загружаем GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_gpt2_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    return response or "Мне сложно ответить 😕"

@dp.message(Command("start"))
async def start_handler(message: types.Message):
    await message.answer("Привет! Я твой GPT-2 AI-бот 🤖\nЗадай мне вопрос!")

@dp.message()
async def ai_answer(message: types.Message):
    user_text = message.text
    answer = generate_gpt2_response(user_text)
    await message.answer(answer)

async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
