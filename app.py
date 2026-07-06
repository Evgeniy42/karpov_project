from datetime import datetime
from typing import List, Dict, Any

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
            f"❌ Ошибка при выполнении SQL-запроса: {e}\nЗапрос: {{query}}"
        ) from e
    finally:
        conn.close()

    return df


# === Загрузка основных ресурсов ===
logger.info("Инициализация сервиса...")

# Создаём объект FastAPI
app = FastAPI()

# Список постов
logger.info("Загружаем список постов...")

df_posts_sorted = load_sql("""SELECT * FROM public.post_text_df ORDER BY post_id""")

logger.success("Список постов успешно загружен.")

logger.success("Сервис успешно инициализирован")


# Эндпойнт для получения рекомендаций
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(user_id: int, dt: datetime, limit: int = 10) -> List[PostGet]:
    """
    Возвращает список рекомендованных постов для пользователя на заданную дату и время.

    Аргументы:
        user_id: идентификатор пользователя
        dt: дата и время запроса (для временных признаков)
        limit: максимальное количество постов в ответе

    Возвращает:
        Список объектов PostGet с топ-N постами
    """

    if user_id % 2 == 0:
        # чётные user_id - берём limit первых постов с чётными post_id
        top_posts = df_posts_sorted[df_posts_sorted['post_id'] % 2 == 0].head(limit)
    else:
        # нечётные user_id → берём limit первых постов с нечётными post_id
        top_posts = df_posts_sorted[df_posts_sorted['post_id'] % 2 == 1].head(limit)

    # Формируем список моделей PostGet
    # TODO: превратите строки датафрейма top_posts в список объектов PostGet
    recs = [
        PostGet(
            id=item['post_id'],
            text=item['text'],
            topic=item['topic']
        )
        for _, item in top_posts.iterrows()
    ]

    return recs
