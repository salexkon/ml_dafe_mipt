# Шаблон проекта HW3

Оформите работу в папке `hw3/` по этой структуре. По коду, конфигурации и отчёту преподаватель должен воспроизвести результат отправленной модели.

```text
hw3/
  configs/
    final.yaml
  src/
    __init__.py
    data.py
    model.py
    train.py
    evaluate.py
    predict.py
  reports/
    report.md
    figures/
  requirements.txt
  .gitignore
```

| Где | Что положить |
|:--|:--|
| `configs/` | Конфигурация итогового запуска и варианты экспериментов. Запишите seed, разбиение, размер входа, тип модели, параметры преобразований и настройки обучения. Для нейросети это могут быть оптимизатор, learning rate, batch size и число эпох. Можно выбрать YAML, TOML или JSON |
| `src/data.py` | Dataset, преобразования, формирование train/validation и соответствие индексов классам |
| `src/model.py` | Архитектура модели |
| `src/train.py` | Обучение и сохранение весов и истории метрик |
| `src/evaluate.py` | Загрузка сохранённых весов и подсчёт метрик на фиксированной валидации |
| `src/predict.py` | Инференс на тестовых изображениях и сохранение CSV для Kaggle |
| `reports/report.md` | Команды запуска, ссылка на итоговые веса, участники и Kaggle-команда, успешные посылки, таблица экспериментов и выводы |
| `reports/figures/` | Графики сравнения экспериментов, матрица ошибок и примеры ошибок |
| `requirements.txt` | Версии зависимостей |
| `.gitignore` | Данные, кеши, промежуточные результаты и файлы с секретами |

В отчёте приведите команды установки зависимостей, обучения, оценки сохранённых весов и получения CSV. Добавьте команду скачивания весов, если они лежат отдельно. Пример формы команд для запуска из `hw3/`:

```bash
python -m pip install -r requirements.txt
python -m src.train --config configs/final.yaml --data data
python -m src.evaluate --config configs/final.yaml --data data --weights models/final.pt
python -m src.predict --config configs/final.yaml --data data --weights models/final.pt --output submission.csv
```

В своём проекте укажите фактические команды и пути. Дополнительные ноутбуки с анализом можно хранить в `notebooks/`. Скрипты итоговой модели должны запускаться в чистом окружении из репозитория.

За основу взято разделение кода, данных, моделей и отчётов. Для HW3 добавлены отдельные точки запуска обучения, оценки и инференса. [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).

Для проектов с большим числом конфигураций есть шаблон с готовыми циклами Lightning и композицией конфигураций Hydra. В HW3 используется описанная выше структура. Библиотеку выбираете сами. [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template).
