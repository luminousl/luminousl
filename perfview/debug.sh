#!/bin/bash

# curl -o- https://fnm.vercel.app/install | bash
# /bin/bash -c "source ~/.bashrc && fnm install 22 && node -v"
# pip install fastapi["standard"] aiofiles onnx requests sqlalchemy pymysql pandas graphviz
# apt update && apt install -y graphviz

export PERFVIEW_DEBUG=1
export PERFVIEW_DATA_ROOT=./datas
export PERFVIEW_REFERENCE_DATA_ROOT=./datas/reference

rm -rf ./datas
rm -rf ./debug.db
/usr/bin/sqlite3 debug.db < product.sql

uvicorn main:app --host=0.0.0.0 --workers=0 --port=8899