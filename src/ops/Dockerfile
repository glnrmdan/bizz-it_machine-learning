FROM python:3.9.16
WORKDIR /ops
COPY ./requirements.txt /ops/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /ops/requirements.txt
COPY ./app /ops/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]