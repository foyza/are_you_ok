from deeppavlov import build_model, configs

# Загружаем модель DeepPavlov Grammar Checker
model = build_model(configs.syntax.syntax_ru, download=True)

# Пример тестовых предложений
examples = [
    "Я иду в магазин",
    "Она идти в магазин",
    "Сегодня хорошая погода",
    "Сегодняхорошая погода",
    "Мы учимся в университете",
    "Мы учится в университете"
]

for sentence in examples:
    corrected = model([sentence])[0]
    changes = sum(1 for a, b in zip(sentence, corrected) if a != b)
    conf = max(0.0, 1 - changes / max(len(sentence), 1))
    conf_percent = round(conf * 100, 1)
    
    if sentence == corrected:
        print(f"✅ '{sentence}' — грамматично, confidence: {conf_percent}%")
    else:
        print(f"❌ '{sentence}' — ошибки, confidence: {conf_percent}%")
        print(f"   Исправлено: {corrected}")
