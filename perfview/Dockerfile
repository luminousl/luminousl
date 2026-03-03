FROM docker.io/python:3.10
LABEL maintainer="jingweid@nvidia.com"

VOLUME /data
VOLUME /reference

ENV PERFVIEW_REFERENCE_DATA_ROOT=/reference
ENV PERFVIEW_DATA_ROOT=/data
ENV PERFVIEW_SERVER_PORT=19220
WORKDIR /app
RUN mkdir -p /app

RUN curl -o- https://fnm.vercel.app/install | bash
RUN /bin/bash -c "source ~/.bashrc && fnm install 22 && node -v"

RUN pip install fastapi["standard"] aiofiles onnx requests sqlalchemy pymysql pandas graphviz
RUN apt update && apt install -y graphviz
COPY . /app/
ENTRYPOINT ["bash", "-c", "-l", "uvicorn main:app --host=0.0.0.0 --workers=16 --port=$PERFVIEW_SERVER_PORT"]