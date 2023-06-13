FROM python:3.9.16
WORKDIR /ops
COPY ./requirements.txt /ops/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /ops/requirements.txt
RUN pip install 'uvicorn[standard]'
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY ./app /ops/app
CMD ["python", "./app/main.py"]
