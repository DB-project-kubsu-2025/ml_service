from fastapi import FastAPI, Response, status

app = FastAPI(
    title='ML Service',
    version='0.1.0',
    docs_url='/swagger',
)


@app.get('/healthcheck')
def healthcheck():
    """Проверка работоспособности сервиса"""
    return Response(status_code=status.HTTP_200_OK)
