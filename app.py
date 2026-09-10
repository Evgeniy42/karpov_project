import os
import pickle
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from loguru import logger

from database import postgres_connection
from schema import PostGet, Response

import hashlib


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


def load_model(model_version: str = "default") -> Any:
    """
    Загружает ML-модель из pickle-файла.

    Если код запускается в LMS-окружении (IS_LMS=1),
    директория берётся из переменной окружения MODELS_DIR.
    Иначе используется текущая директория.

    Аргументы:
        model_version: версия модели для загрузки (имя файла без префикса model_ и расширения .pkl)

    Исключения:
        FileNotFoundError — если файл модели не найден.
        RuntimeError — если произошла ошибка при загрузке модели.
    """
    # Определяем директорию с моделями
    if os.environ.get("IS_LMS", "0") == "1":
        models_dir = os.environ.get("MODELS_DIR", ".")
    else:
        models_dir = "model"

    # Формируем путь к файлу модели с префиксом model_
    model_filename = f"model_{model_version}.pkl"
    model_path = os.path.join(models_dir, model_filename)

    logger.info(f"Загрузка модели версии '{model_version}' из файла {model_path}...")

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Файл модели версии '{model_version}' не найден: {model_path}")
    except Exception as e:
        raise RuntimeError(f"❌ Ошибка при загрузке модели версии '{model_version}': {e}") from e

    logger.success(f"Модель версии '{model_version}' успешно загружена")

    return model


# === Загрузка основных ресурсов ===
logger.info("Инициализация сервиса...")

# Создаём объект FastAPI
app = FastAPI()

# Загружаем модель в память
model_test = load_model(model_version='test')
model_control = load_model(model_version='control')

# Фичи для предсказания
user_features = load_sql('''SELECT * FROM "public"."evgenij-bulatov-jta6567_user_features"''')
post_features = load_sql('''SELECT * FROM "public"."evgenij-bulatov-jta6567_post_features"''')

# Посты в обычном виде
post_info = load_sql('''SELECT * FROM public.post_text_df''')

# Соль для разбиения A/B пользователей
salt_ab = "salt_for_testing"

# Список фичей в правильном порядке, в котором модель обучалась
test_feature_columns = ['user_id',
                     'post_id',
                     'hour',
                     'age',
                     'exp_group',
                     'mean_tfidf_like_posts',
                     'mean_emb_like_posts',
                     'emb_angle_30_all',
                     'like_cluster',
                     'stability',
                     'mean_city',
                     'lenght_post',
                     'tfidf_mean',
                     'tfidf_max',
                     'svd_column',
                     'rbf_centr_1',
                     'rbf_centr_2',
                     'rbf_centr_3',
                     'rbf_centr_4',
                     'rbf_centr_5',
                     'rbf_centr_7',
                     'rbf_centr_8',
                     'rbf_centr_9',
                     'rbf_centr_10',
                     'mean_emb',
                     'max_emb',
                     'like_count',
                     'rbf_centr_emb_1',
                     'rbf_centr_emb_2',
                     'rbf_centr_emb_3',
                     'rbf_centr_emb_4',
                     'rbf_centr_emb_5',
                     'rbf_centr_emb_6',
                     'rbf_centr_emb_7',
                     'rbf_centr_emb_8',
                     'rbf_centr_emb_9',
                     'rbf_centr_emb_10'
]


control_feature_columns = ['OneHot__os_iOS', 'OneHot__source_organic', 'MeanTarget__country',
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


# Разбиение пользователей по группам
def get_exp_group(user_id: int) -> str:

    comb_str = f'{salt_ab}_{str(user_id)}'
    hash = hashlib.md5(comb_str.encode('utf-8'))

    hash_int = int(hash.hexdigest(), 16)

    group_num = hash_int % 2

    if group_num == 1:
        group_user = 'control'
    else:
        group_user = 'test'

    return group_user


# Эндпойнт для получения рекомендаций (Новая тестовая модель)
@app.get("/post/test_recommendations/", response_model=List[PostGet])
def test_recommended_posts(user_id: int, dt: datetime, limit: int = 10) -> List[PostGet]:
    """
    Возвращает список рекомендованных постов для пользователя на заданную дату и время.
    """
    # В этом эндпойнте мы используем ранее загруженную модель и признаки.

    # Временные признаки
    hour = int(dt.hour)

    # Формирование данных для рекомендации
    df = post_features.sample(frac=0.5, random_state=42)

    # df = post_features.copy()
    df['user_id'] = user_id

    user_row = user_features[user_features['user_id'] == user_id].iloc[0]
    df['mean_city'] = user_row['mean_city']
    df['age'] = user_row['age']
    df['exp_group'] = user_row['exp_group']
    df['mean_tfidf_like_posts'] = user_row['mean_tfidf_like_posts']
    df['mean_emb_like_posts'] = user_row['mean_emb_like_posts']
    df['emb_angle_30_all'] = user_row['emb_angle_30_all']
    df['like_cluster'] = user_row['like_cluster']
    df['stability'] = user_row['stability']

    df['hour'] = hour

    # Формирование данных в правильный порядок
    df = df[test_feature_columns]

    # Получаем предсказания в виде вероятностей
    df['prediction'] = model_test.predict_proba(df)[:, 1]

    # Формируем список limit постов, где самая высокая вероятность того, что пользователю понравится пост.
    top_post_idx = df.nlargest(limit, 'prediction')['post_id'].tolist()

    # Получаем рекомендованные посты в обычном виде
    recs_posts = post_info[post_info['post_id'].isin(top_post_idx)]

    recs = [
        PostGet(id=item.post_id, text=item.text, topic=item.topic)
        for item in recs_posts.itertuples()
    ]

    return recs


# Эндпойнт для получения рекомендаций (Контрольная первая модель)
@app.get("/post/control_recommendations/", response_model=List[PostGet])
def control_recommended_posts(user_id: int, dt: datetime, limit: int = 10) -> List[PostGet]:
    """
    Возвращает список рекомендованных постов для пользователя на заданную дату и время.
    """
    # В этом эндпойнте мы используем ранее загруженную модель и признаки.

    # Временные признаки
    month = int(dt.month)
    day = int(dt.day)
    hour = int(dt.hour)

    # Формирование данных для рекомендации
    df = post_features.sample(frac=0.5, random_state=42)

    # df = post_features.copy()
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
    df = df[control_feature_columns]

    # Получаем предсказания в виде вероятностей
    df['prediction'] = model_control.predict_proba(df)[:, 1]

    # Формируем список limit постов, где самая высокая вероятность того, что пользователю понравится пост.
    top_post_idx = df.nlargest(limit, 'prediction')['post_id'].tolist()

    # Получаем рекомендованные посты в обычном виде
    recs_posts = post_info[post_info['post_id'].isin(top_post_idx)]

    recs = [
        PostGet(id=item.post_id, text=item.text, topic=item.topic)
        for item in recs_posts.itertuples()
    ]

    return recs


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(user_id: int, dt: datetime, limit: int = 10) -> Response:

    group_user = get_exp_group(user_id)

    logger.info(f'Пользователь {user_id} попал в контрольную группу: {group_user}')

    if group_user == 'control':
        recs = control_recommended_posts(user_id, dt, limit)

        return Response(
            exp_group=group_user,
            recommendations=recs
        )

    else:
        recs = test_recommended_posts(user_id, dt, limit)

        return Response(
            exp_group=group_user,
            recommendations=recs
        )
