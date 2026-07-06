import os
import pickle
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from loguru import logger

from database import postgres_connection
from schema import PostGet


# === Вспомогательные функции ===
def load_sql(query: str, dtypes: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Выполняет SQL-запрос через соединение postgres_connection и возвращает DataFrame.

    Аргументы:
        query: SQL-запрос
        dtypes: словарь типов колонок для pd.read_sql (по умолчанию None)

    Возвращает:
        pd.DataFrame с результатом запроса

    Исключения:
        RuntimeError, если произошла ошибка при выполнении запроса
    """
    conn = postgres_connection()

    try:
        df = pd.read_sql(query, conn, dtype=dtypes)
    except Exception as e:
        raise RuntimeError(
            f"❌ Ошибка при выполнении SQL-запроса: {e}\nЗапрос: {query}"
        ) from e
    finally:
        conn.close()

    return df


def load_model(model_path: str = "model.pkl"):
    """
    Загружает ML-модель из pickle-файла.

    Если код запускается в LMS-окружении (IS_LMS=1),
    путь берётся из переменной окружения MODEL_PATH.
    Иначе используется локальный путь, переданный пользователем.

    Исключения:
        FileNotFoundError — если файл модели не найден.
        RuntimeError — если произошла ошибка при загрузке модели.
    """
    if os.environ.get("IS_LMS", "0") == "1":
        model_path = os.environ["MODEL_PATH"]

    logger.info(f"Загрузка модели из файла {model_path}...")

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Файл модели не найден: {model_path}")
    except Exception as e:
        raise RuntimeError(f"❌ Ошибка при загрузке модели: {e}") from e

    logger.success("Модель успешно загружена")

    return model


# === Загрузка основных ресурсов ===
logger.info("Инициализация сервиса...")

# Создаём объект FastAPI
app = FastAPI()

# Загружаем модель в память
model = load_model(model_path='model/recommendation_model.pkl')

# Фичи для предсказания
user_features = load_sql('''SELECT * FROM "public"."evgenij-bulatov-jta6567_user_features"''')
post_features = load_sql('''SELECT * FROM "public"."evgenij-bulatov-jta6567_post_features"''')

# Посты в обычном виде
post_info = load_sql('''SELECT * FROM public.post_text_df''')

# Список фичей в правильном порядке, в котором модель обучалась
feature_columns = ['OneHot__os_iOS', 'OneHot__source_organic', 'MeanTarget__country',
       'MeanTarget__city', 'MeanTarget__topic', 'month',
       'day', 'hour', 'user_id',
       'post_id', 'gender', 'age',
       'exp_group', 'mean_tfidf_like_posts',
       'lenght_post', 'tfidf_mean',
       'tfidf_max', 'svd_column',
       'rbf_centr_1', 'rbf_centr_2',
       'rbf_centr_3', 'rbf_centr_4',
       'rbf_centr_5', 'rbf_centr_6',
       'rbf_centr_7', 'rbf_centr_8',
       'rbf_centr_9', 'rbf_centr_10',
       'favorite_topic']

logger.success("Сервис успешно инициализирован")


# Эндпойнт для получения рекомендаций
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(user_id: int, dt: datetime, limit: int = 10) -> List[PostGet]:
    """
    Возвращает список рекомендованных постов для пользователя на заданную дату и время.
    """
    # В этом эндпойнте мы используем ранее загруженную модель и признаки.

    # Временные признаки
    month = int(dt.month)
    day = int(dt.day)
    hour = int(dt.hour)

    # Формирование данных для рекомендации
    df = post_features.copy()
    df['user_id'] = user_id

    user_row = user_features[user_features['user_id'] == user_id].iloc[0]
    df['OneHot__os_iOS'] = user_row['OneHot__os_iOS']
    df['OneHot__source_organic'] = user_row['OneHot__source_organic']
    df['MeanTarget__country'] = user_row['MeanTarget__country']
    df['MeanTarget__city'] = user_row['MeanTarget__city']
    df['gender'] = user_row['gender']
    df['age'] = user_row['age']
    df['exp_group'] = user_row['exp_group']
    df['mean_tfidf_like_posts'] = user_row['mean_tfidf_like_posts']

    df['favorite_topic'] = 0

    df['month'] = month
    df['day'] = day
    df['hour'] = hour

    # Формирование данных в правильный порядок
    df = df[feature_columns]

    # Получаем предсказания в виде вероятностей
    df['prediction'] = model.predict_proba(df)[:, 1]

    # Формируем список limit постов, где самая высокая вероятность того, что пользователю понравится пост.
    top_post_idx = df.nlargest(limit, 'prediction')['post_id'].tolist()

    # Получаем рекомендованные посты в обычном виде
    recs_posts = post_info[post_info['post_id'].isin(top_post_idx)]

    recs = [
        PostGet(id=item.post_id, text=item.text, topic=item.topic)
        for item in recs_posts.itertuples()
    ]

    return recs
