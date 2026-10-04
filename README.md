# Iris ML Production

Учебный ML-проект по классификации цветков Iris, переработанный из простого Python-скрипта в production-like структуру.

## Задача

Модель классифицирует цветок Iris по четырём признакам:

- sepal length
- sepal width
- petal length
- petal width

Поддерживаются три класса:

- setosa
- versicolor
- virginica

## Структура проекта

iris-ml-production/
├── iris_classifier/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   └── predict.py
├── tests/
│   ├── test_data.py
│   └── test_predict.py
├── models/
│   └── .gitkeep
├── .gitignore
├── .pre-commit-config.yaml
├── poetry.lock
├── pyproject.toml
└── README.md

## Технологии

- Python 3.13
- scikit-learn
- Poetry
- pytest
- Ruff
- pre-commit
- Git

## Установка

Клонирование репозитория:

git clone https://github.com/yasinnsbts/iris-ml-production.git
cd iris-ml-production

Установка зависимостей:

poetry install

Установка pre-commit hook:

poetry run pre-commit install

## Обучение модели

poetry run python -m iris_classifier.train

Обученная модель сохраняется в:

models/iris_model.joblib

Файл модели является генерируемым артефактом и не хранится в Git.

## Предсказание

poetry run python -m iris_classifier.predict

Пример результата:

Predicted species: setosa

## Тестирование

poetry run pytest -q

## Проверка качества кода

Линтинг:

poetry run ruff check .

Проверка форматирования:

poetry run ruff format --check .

Все pre-commit проверки:

poetry run pre-commit run --all-files

## Рефакторинг

Исходная версия проекта состояла из одного файла train.py, содержащего загрузку данных, обучение модели и оценку качества.

После рефакторинга:

- код разделён на отдельный Python-пакет;
- загрузка данных отделена от обучения;
- обучение отделено от inference;
- используется sklearn Pipeline;
- добавлено логирование;
- модель сохраняется как отдельный артефакт;
- зависимости управляются через Poetry;
- виртуальное окружение изолировано и исключено из Git;
- добавлены автоматические тесты;
- настроены Ruff и pre-commit.