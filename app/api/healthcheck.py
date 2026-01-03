from fastapi import APIRouter, Response, status

router = APIRouter()


@router.get('/healthcheck')
def healthcheck():
    """Проверка работоспособности сервиса"""
    return Response(status_code=status.HTTP_200_OK)
