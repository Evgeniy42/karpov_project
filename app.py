from typing import List
from fastapi import FastAPI, HTTPException, Depends
from database import postgres_connection
from schema import UserGet, PostGet, FeedGet
from helpers import get_user, get_post, get_feed, get_recommended_feed

# Инициализация FastAPI-приложения — точка входа для всех маршрутов
app = FastAPI()


# Функция для создания и закрытия подключения к базе данных.
# Используется в Depends — FastAPI сам управляет подключением и закрытием.
def get_conn():
    """
    Создаёт и возвращает подключение к PostgreSQL через psycopg2.

    Подключение автоматически закрывается после выполнения запроса.
    Используется как зависимость FastAPI.
    """
    conn = postgres_connection()
    try:
        yield conn  # отдаём соединение в обработчик запроса
    finally:
        conn.close()  # обязательно закрываем соединение после завершения


@app.get("/user/{id}", response_model=UserGet)
def handle_get_user(id: int, conn=Depends(get_conn)) -> UserGet:
    """
    Получить информацию о пользователе по ID.

    Параметры:
        id (int): Уникальный идентификатор пользователя.

    Возвращает:
        UserGet: Данные пользователя в формате, пригодном для API.

    Исключения:
        HTTPException 404 — если пользователь с заданным ID не найден.
    """
    user = get_user(conn, id)
    if user is not None:
        return UserGet(**user.__dict__)
    raise HTTPException(404, f"User with id={id} not found")


@app.get("/post/{id}", response_model=PostGet)
def handle_get_post(id: int, conn=Depends(get_conn)) -> PostGet:
    """
    Получить информацию о посте по его ID.

    Параметры:
        id (int): Уникальный идентификатор поста.

    Возвращает:
        PostGet: Информация о посте (текст и тема).

    Исключения:
        HTTPException 404 — если пост с заданным ID не найден.
    """
    post = get_post(conn, id)
    if post is not None:
        return PostGet(**post.__dict__)
    raise HTTPException(404, f"Post with id={id} not found")


@app.get("/user/{id}/feed", response_model=List[FeedGet])
def handle_get_user_feed(
    id: int, limit: int = 10, conn=Depends(get_conn)
) -> List[FeedGet]:
    """
    Получить список действий пользователя (лайки, просмотры) по его ID.

    Параметры:
        id (int): Идентификатор пользователя.
        limit (int): Максимальное количество действий в ответе (по умолчанию 10).

    Возвращает:
        List[FeedGet]: Список действий, отсортированных от новых к старым.
    """
    list_feed = get_feed(conn, user_id=id, limit=limit)
    for row in list_feed:
        row.user = UserGet(**row.user.__dict__)
        row.post = PostGet(**row.post.__dict__)
    return [FeedGet(**row.__dict__) for row in list_feed]


@app.get("/post/{id}/feed", response_model=List[FeedGet])
def handle_get_post_feed(
    id: int, limit: int = 10, conn=Depends(get_conn)
) -> List[FeedGet]:
    """
    Получить список действий пользователей с заданным постом.

    Параметры:
        id (int): Идентификатор поста.
        limit (int): Максимальное количество действий (по умолчанию 10).

    Возвращает:
        List[FeedGet]: Список действий пользователей с этим постом,
        отсортированный от новых к старым.
    """
    list_feed = get_feed(conn, post_id=id, limit=limit)
    for row in list_feed:
        row.user = UserGet(**row.user.__dict__)
        row.post = PostGet(**row.post.__dict__)
    return [FeedGet(**row.__dict__) for row in list_feed]


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
    id: int, limit: int = 10, conn=Depends(get_conn)
) -> List[PostGet]:
    """
    Получить рекомендованные посты (baseline-версия).

    Возвращает топ-N популярных постов по количеству лайков.
    Это базовая версия рекомендательной системы, одинаковая для всех пользователей.

    Параметры:
        id (int): ID пользователя (пока не используется, зарезервирован для будущей персонализации).
        limit (int): Количество постов в выдаче (по умолчанию 10).

    Возвращает:
        List[PostGet]: Список популярных постов, отсортированных по убыванию лайков.
    """
    list_post = get_recommended_feed(conn, id, limit=limit)
    return [PostGet(**row.__dict__) for row in list_post]
