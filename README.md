# ml_service

## Системные требования
- python 3.12+
- Docker

## Настройка тестового окружения для разработки
- Создать виртуальное окружение python
```bash
python -m venv venv
```
- Активировать его 
```bash 
.\venv\Scripts\activate
```
- Установить poetry
```bash
python -m pip install poetry==2.1.3
```
- Установить зависимости
```bash
poetry install --no-root
```
- создать docker-network (если ранее не была создана)
```bash
docker network create db_project_network
```
- запустить приложение
```bash
docker compose up
```
- по адресу `<ip-адрес>:8071/swagger` можно проверить, что все работает как надо

## Настройка интерпретатора для запуска через PyCharm
- В настройках (Ctrl + Alt + S) найти `Project ml_service` и перейти в `Python interpreter`
- Нажать `Add interpreter` -> `on docker compose`
- Прокликать оставшиеся кнопки, не меняя никаких настроек, дождаться настройки интерпретатора
- Справа сверху PyCharm-а настроить конфигурацию запуска, выбрав `fastapi` и в `Python interpreter` указать созданный интерпретатор в списке
