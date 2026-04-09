from typing import List, Optional
from psycopg2.extensions import connection
from psycopg2.extras import DictCursor

from models import User, Post, Feed


def get_user(conn: connection, user_id: int) -> Optional[User]:
    """
    Загружает одного пользователя из базы данных по его Id.

    Возвращает объект User, если пользователь найден.
    Если пользователь с таким id отсутствует — возвращает None.
    """
    query = """
    SELECT *
    FROM public.user
    WHERE id = %s
    """

    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query, (user_id,))
        row = cur.fetchone()

    if row:
        return User(
            id=row['id'],
            gender=row['gender'],
            age=row['age'],
            country=row['country'],
            city=row['city'],
            exp_group=row['exp_group'],
            os=row['os'],
            source=row['source']
        )
    return None


def get_post(conn: connection, post_id: int) -> Optional[Post]:
    """
    Загружает один пост из базы данных по его Id.

    Возвращает объект Post, если пост найден.
    Если пост с таким id отсутствует — возвращает None.
    """
    query = """
    SELECT *
    FROM public.post
    WHERE id = %s
    """

    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query, (post_id,))
        row = cur.fetchone()

    if row:
        return Post(
            id=row['id'],
            text=row['text'],
            topic=row['topic']
        )
    return None


def get_feed(
    conn: connection, user_id: int = None, post_id: int = None, limit: int = 10
) -> List[Feed]:
    """
    Получает список действий пользователей с постами, включая данные о пользователях и постах.

    - Необходимо указать хотя бы один фильтр: user_id или post_id.
    - Возвращает не более `limit` записей.
    - Действия сортируются по времени: от самых свежих к более старым.
    - Используется для получения последних активностей пользователя или взаимодействий с постом.
    """
    if user_id is None and post_id is None:
        raise ValueError("Необходимо указать хотя бы user_id или post_id")

    # - выбрать данные из feed_action
    # - JOIN с user и post
    # - фильтрация по user_id / post_id (если заданы)
    # - сортировка по времени (DESC)
    # - ограничение по LIMIT
    query = """
    SELECT u.id as user_id, u.gender, u.age, u.country, u.city, u.exp_group, u.os, u.source, p.id as post_id,
    p.text, p.topic, f.action, f.time
    FROM public.feed_action as f
    INNER JOIN public.user as u ON f.user_id = u.id
    INNER JOIN public.post as p ON f.post_id = p.id
    WHERE (%s IS NULL OR f.user_id = %s)
    AND (%s IS NULL OR f.post_id = %s)
    ORDER BY time DESC
    LIMIT %s
    """

    result = []
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query, (user_id, user_id, post_id, post_id, limit))
        data = cur.fetchall()

        for row in data:
            result.append(Feed(
                user_id=row['user_id'],
                post_id=row['post_id'],
                user=User(
                    id=row['user_id'],
                    gender=row['gender'],
                    age=row['age'],
                    country=row['country'],
                    city=row['city'],
                    exp_group=row['exp_group'],
                    os=row['os'],
                    source=row['source']
                ),
                post=Post(
                    id=row['post_id'],
                    text=row['text'],
                    topic=row['topic']
                ),
                action=row['action'],
                time=row['time']
            ))

    return result


def get_recommended_feed(conn: connection, id: int, limit: int) -> List[Post]:
    """
    Возвращает список top-N постов с наибольшим числом лайков.

    Это базовая реализация рекомендательной системы (baseline),
    которая не учитывает индивидуальные предпочтения, а показывает
    одинаковые популярные посты всем пользователям.

    Параметры:
        conn (connection): подключение к базе данных.
        id (int): ID пользователя (в этой версии не используется,
                  но оставлен для совместимости с будущей логикой).
        limit (int): количество постов в выдаче.

    Возвращает:
        List[Post]: список объектов Post, отсортированных по убыванию популярности.
    """
    query = """
    SELECT p.id, p.text, p.topic, COUNT(f.user_id)
    FROM public.feed_action f
    JOIN public.post p ON f.post_id = p.id
    WHERE f.action = 'like'
    GROUP BY p.id
    ORDER BY COUNT(f.user_id) DESC
    LIMIT %s
    """

    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query, (limit,))
        rows = cur.fetchall()

    return [Post(
        id=row['id'],
        text=row['text'],
        topic=row['topic']
    ) for row in rows]
