#!/bin/bash

# MySQL
# export PERFVIEW_DATA_ROOT=/datav/onnx_views_data/product_data
# export PERFVIEW_REFERENCE_DATA_ROOT=/datav/files_storage
# export PERFVIEW_DB_NAME=trt_perf_view
# export PERFVIEW_DB_USER=root
# export PERFVIEW_DB_PASSWORD=nvcntse
# export PERFVIEW_DB_HOST=127.0.0.1
# export PERFVIEW_DB_PORT=3306
# export PERFVIEW_DB_CHARSET=utf8mb3

# Sqlite
export PERFVIEW_DEBUG=1
export PERFVIEW_DATA_ROOT=./product_datas
export PERFVIEW_REFERENCE_DATA_ROOT=./product_datas/reference
export PERFVIEW_DATABASE_FILE=./product.db

uvicorn main:app --host=0.0.0.0 --workers=8 --port=8822