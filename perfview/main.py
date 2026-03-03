import fastapi
from fastapi import Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse, RedirectResponse
import os
import json
import aiofiles
from onnx_layout_generate import onnx_layout_generate, dump_onnx
import requests
import traceback
from sqlalchemy.orm import Session
from db.session import default_db_depends
from db import dao, schemas
from urllib import parse as urlparse
from kernel_match_report import generate_kernel_match_report
from generate_from_logfile import parse_layer_info_and_performance_and_model
import onnx
import re
import hashlib
import argparse
from io import BytesIO
from datetime import datetime, timedelta
import pandas as pd
from dot_dag import generate_graph_svg
import urllib3
import shutil
from tempfile import NamedTemporaryFile
import time
from create_view import signature_file
from health_check import HealthChecker
from generate_onnx_from_code import generate_onnx_from_code
from profiling_pipeline import ProfilingPipeline
from multiprocessing import Process, Queue
import threading
from queue import Queue

urllib3.disable_warnings()

DATA_ROOT = os.path.abspath(os.getenv("PERFVIEW_DATA_ROOT", "./datas"))
REFERENCE_DATA_ROOT = os.path.abspath(os.getenv("PERFVIEW_REFERENCE_DATA_ROOT", "./datas/reference"))
ONNX_LAYOUT_CACHING_ROOT = os.path.abspath(os.path.join(DATA_ROOT, "caching", "onnx_layout"))
if not os.path.exists(DATA_ROOT):
    os.makedirs(DATA_ROOT, exist_ok=True)

if not os.path.exists(REFERENCE_DATA_ROOT):
    os.makedirs(REFERENCE_DATA_ROOT, exist_ok=True)

if not os.path.exists(ONNX_LAYOUT_CACHING_ROOT):
    os.makedirs(ONNX_LAYOUT_CACHING_ROOT, exist_ok=True)

TEMP_DIR = os.path.join(DATA_ROOT, "temp")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR, exist_ok=True)

print(f"REFERENCE_DATA_ROOT: {REFERENCE_DATA_ROOT}")
print(f"DATA_ROOT: {DATA_ROOT}")
print(f"TEMP_DIR: {TEMP_DIR}")

class AsyncPipelineRunner(threading.Thread):
    def __init__(self):
        super().__init__()
        self.task_queue = Queue()

    def run_pipeline(self, view_id, onnx_path, code, profile_name, 
        status_file,
        view_folder,
        profiling_folder, profiling_script
    ):
        meta_file = os.path.join(view_folder, "subgraph_meta.json")
        subgraph_meta = {}
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                subgraph_meta = json.load(f)

        if profile_name not in subgraph_meta:
            subgraph_meta[profile_name] = dict(
                code = code,
                create_time = format_date(datetime.now())
            )
        else:
            subgraph_meta[profile_name]["code"] = code
            subgraph_meta[profile_name]["update_time"] = format_date(datetime.now())
        
        with open(status_file, "w") as f:
            json.dump({"status": "running", "message": "Compiling the subgraph code..."}, f)

        profile = subgraph_meta[profile_name]
        model_proto = onnx.load(onnx_path, load_external_data=False)
        output = generate_onnx_from_code(profile["code"], model_proto)
        if output["status"] != "success":
            with open(status_file, "w") as f:
                json.dump({"status": "error", "message": "Failed to compile the code: " + output["message"]}, f)
            return

        profiling_model_path = os.path.join(profiling_folder, "model.onnx")
        model = output["model"]
        del model_proto
        onnx.save_model(model, profiling_model_path)

        attached_files = profile.get("attached_files", [])
        attached_files_full_path = [profiling_model_path]
        for file in attached_files:
            attached_files_full_path.append(os.path.join(profiling_folder, "inputs", file["name"]))

        with open(status_file, "w") as f:
            json.dump({"status": "running", "message": "Pipeline run.."}, f)

        def state_update_fn(message):
            with open(status_file, "w") as f:
                json.dump({"status": "running", "message": message}, f)

        pipeline = ProfilingPipeline(profiling_script, attached_files_full_path, profiling_folder, TEMP_DIR)
        profiling_result = pipeline.run(state_update_fn)
        profile["profiling_files"] = profiling_result
        profile["profiling_time"]  = format_date(datetime.now())
        profile["profiling_script"] = profiling_script
        with open(meta_file, "w") as f:
            json.dump(subgraph_meta, f)

        with open(status_file, "w") as f:
            json.dump({"status": "success", "files": profiling_result}, f)

    def stop(self):
        self.task_queue.put(dict(cmd="stop"))
        self.join()

    def run(self):
        while True:
            task = self.task_queue.get()
            if task["cmd"] == "stop":
                break

            del task["cmd"]

            status_file = task["status_file"]
            try:
                self.run_pipeline(**task)
            except Exception as e:
                traceback.print_exc()
                print(f"Catch a exception: {e}, status_file: {status_file}")
                with open(status_file, "w") as f:
                    json.dump({"status": "error", "message": str(e)}, f)
        
    def commit(self, **kwargs):
        kwargs["cmd"] = "task"
        self.task_queue.put(kwargs)

app = fastapi.FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/datas",  StaticFiles(directory=DATA_ROOT), name="datas")
async_pipeline_runner = AsyncPipelineRunner()

@app.on_event("startup")
async def shutdown_event():
    async_pipeline_runner.start()
    print("Start async pipeline runner")

@app.on_event("shutdown")
async def shutdown_event():
    async_pipeline_runner.stop()
    print("Cleaning up resources...")

def format_date(date:datetime):
    return (date + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")

@app.get("/")
def show():
    return FileResponse("static/index.html")

@app.get("/views")
def show():
    return FileResponse("static/views.html")

@app.get("/editor")
def show():
    return FileResponse("static/editor.html")

def success(**obj):
    return JSONResponse(dict(status="ok", data=obj))

def failed(msg, **kwargs):
    return JSONResponse(dict(status="failed", message=msg, **kwargs))

@app.post("/create_file_from_local_path")
async def create_file_from_local_path(request: Request, db: Session = default_db_depends):
    params = await request.json()
    if params is None:
        return failed("Missing the required parameters")
    
    path = params.get("path", None)
    if path is None:
        return failed("Missing the required parameter: path")
    
    if path.find("..") != -1:
        return failed(f"Invalid local_path: {path}")
    
    abs_local_path = os.path.join(REFERENCE_DATA_ROOT, path)
    if not os.path.exists(abs_local_path):
        return failed(f"Not found the local file: {abs_local_path}")

    CHUNK_SIZE = 1024 * 64
    signaturer = hashlib.md5()
    with open(abs_local_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if len(chunk) == 0:
                break
            signaturer.update(chunk)

    total_file_size = os.path.getsize(abs_local_path)
    file_name = urlparse.unquote(request.headers.get("file_name", os.path.basename(abs_local_path)))
    folder = urlparse.unquote(request.headers.get("folder", "default"))
    description = urlparse.unquote(request.headers.get("description", ""))
    storage_folder  = os.path.join(DATA_ROOT, "files")
    os.makedirs(storage_folder, exist_ok=True)
    
    instance = dao.create_file(db, schemas.File(
        virtual_folder=folder,
        name=file_name,
        signature=signaturer.hexdigest() + f"{total_file_size:016d}",
        description=description,
        size_bytes=total_file_size
    ))
    file_id = instance.id
    path = os.path.join(storage_folder, f"{file_id:012d}")
    # If the link file already exists, remove it first
    if os.path.exists(path):
        os.remove(path)
    os.symlink(abs_local_path, path)
    db.commit()
    return success(file=dict(
        file_id=instance.id,
        file_name=instance.name,
        size_bytes=instance.size_bytes,
        create_time=format_date(instance.create_time),
        update_time=format_date(instance.update_time),
        virtual_folder=instance.virtual_folder,
        description=instance.description
    ))

@app.post("/create_file_by_upload")
async def create_file_by_upload(request: Request, file: UploadFile, db: Session = default_db_depends):
    if file is None:
        return failed("unknow item file")
    
    total_file_size = 0
    CHUNK_SIZE = 1024 * 16
    signaturer = hashlib.md5()
    with NamedTemporaryFile(delete=False, dir=TEMP_DIR) as tf:
        async with aiofiles.open(tf.name, 'wb') as f:
            while chunk := await file.read(CHUNK_SIZE):
                total_file_size += len(chunk)
                signaturer.update(chunk)
                await f.write(chunk)

    file_name = urlparse.unquote(request.headers.get("file_name", file.filename))
    folder = urlparse.unquote(request.headers.get("folder", "default"))
    description = urlparse.unquote(request.headers.get("description", ""))
    storage_folder  = os.path.join(DATA_ROOT, "files")
    os.makedirs(storage_folder, exist_ok=True)
    
    instance = dao.create_file(db, schemas.File(
        virtual_folder=folder,
        name=file_name,
        signature=signaturer.hexdigest() + f"{total_file_size:016d}",
        description=description,
        size_bytes=total_file_size
    ))
    file_id = instance.id
    path = os.path.join(storage_folder, f"{file_id:012d}")
    shutil.move(tf.name, path)
    db.commit()
    return success(file=dict(
        file_id=instance.id,
        file_name=instance.name,
        size_bytes=instance.size_bytes,
        create_time=format_date(instance.create_time),
        update_time=format_date(instance.update_time),
        virtual_folder=instance.virtual_folder,
        description=instance.description
    ))
    
@app.post("/create_file_by_upload_directio")
async def create_file_by_upload_directio(request: Request, db: Session = default_db_depends):

    total_file_size = 0
    signaturer = hashlib.md5()
    with NamedTemporaryFile(delete=False, dir=TEMP_DIR) as tf:
        async with aiofiles.open(tf.name, 'wb') as f:
            async for chunk in request.stream():
                total_file_size += len(chunk)
                signaturer.update(chunk)
                await f.write(chunk)
    
    storage_folder  = os.path.join(DATA_ROOT, "files")
    os.makedirs(storage_folder, exist_ok=True)
    file_name = urlparse.unquote(request.headers.get("file_name", ""))
    folder = urlparse.unquote(request.headers.get("folder", "default"))
    description = urlparse.unquote(request.headers.get("description", ""))
    instance = dao.create_file(db, schemas.File(
        virtual_folder=folder,
        name=file_name,
        signature=signaturer.hexdigest() + f"{total_file_size:016d}",
        description=description,
        size_bytes=total_file_size
    ))
    file_id = instance.id
    path = os.path.join(storage_folder, f"{file_id:012d}")
    shutil.move(tf.name, path)
    db.commit()
    return success(file=dict(
        file_id=instance.id,
        file_name=instance.name,
        size_bytes=instance.size_bytes,
        create_time=format_date(instance.create_time),
        update_time=format_date(instance.update_time),
        virtual_folder=instance.virtual_folder,
        description=instance.description
    ))

def get_file_instance(db:Session, file_id):
    if file_id is None or not isinstance(file_id, int):
        return None
    
    path = os.path.join(DATA_ROOT, "files", f"{file_id:012d}")
    if not os.path.exists(path):
        print(f"Not exist path: {path}")
        return None
    
    file_instance = dao.find_file_by_id(db, file_id)
    return path, file_instance

def load_normal_file(db:Session, store, key, file_id):
    try:
        file = get_file_instance(db, file_id)
        if file is None:
            # print(f"Failed to get_file_instance for: {file_id}")
            return None

        path, file_instance = file
        store[f"{key}_meta"] = dict(
            file_id=file_instance.id,
            name=file_instance.name,
            description=file_instance.description,
            signature=file_instance.signature,
            size_bytes=file_instance.size_bytes,
            virtual_folder=file_instance.virtual_folder,
            create_time=format_date(file_instance.create_time),
            update_time=format_date(file_instance.update_time),
        )
        return path, file_instance
    except Exception as e:
        traceback.print_exception(e)
        db.rollback()
    return None

def load_json_file(db:Session, store, key, file_id):
    try:
        file = get_file_instance(db, file_id)
        if file is None:
            # print(f"Failed to get_file_instance for: {file_id}")
            return None

        path, file_instance = file
        with open(path, "r") as f:
            store[key] = json.load(f)
            store[f"{key}_meta"] = dict(
                file_id=file_instance.id,
                name=file_instance.name,
                description=file_instance.description,
                signature=file_instance.signature,
                size_bytes=file_instance.size_bytes,
                virtual_folder=file_instance.virtual_folder,
                create_time=format_date(file_instance.create_time),
                update_time=format_date(file_instance.update_time),
            )
        return file_instance
    except Exception as e:
        traceback.print_exception(e)
        db.rollback()
    return None

def load_csv_file(db:Session, store, key, file_id):
    try:
        file = get_file_instance(db, file_id)
        if file is None:
            # print(f"Failed to get_file_instance for: {file_id}")
            return None

        path, file_instance = file
        with open(path, "r") as f:
            store[key] = json.loads(pd.read_csv(f).to_json(orient="split"))
            store[f"{key}_meta"] = dict(
                file_id=file_instance.id,
                name=file_instance.name,
                description=file_instance.description,
                signature=file_instance.signature,
                size_bytes=file_instance.size_bytes,
                virtual_folder=file_instance.virtual_folder,
                create_time=format_date(file_instance.create_time),
                update_time=format_date(file_instance.update_time),
            )
        return file_instance
    except Exception as e:
        traceback.print_exception(e)
        db.rollback()
    return None

def preprocess_layerinfo_json(layerinfo):
    if layerinfo is None or "Layers" not in layerinfo:
        return layerinfo
    
    onnx_name_parser = re.compile(r"\[ONNX Layer: ([^\]]+)\]")
    myelin_parser = re.compile(r"\{ForeignNode\[(.*?)\.\.\.(.*?)\]\}")
    layers = layerinfo["Layers"]
    for layer in layers:
        onnx_names = []
        if "Metadata" in layer:
            metadata = layer["Metadata"]
            onnx_names = onnx_name_parser.findall(metadata)
            del layer["Metadata"]
        
        if len(onnx_names) == 0 and "LayerType" in layer and "Name" in layer and layer["Name"].find("{ForeignNode[") != -1:
            myelin_names = myelin_parser.findall(layer["Name"])
            if len(myelin_names) > 0:
                onnx_names = list(myelin_names[0])
        
        if len(onnx_names) == 0 and "Name" in layer:
            layer_name = layer["Name"]
            onnx_names = layer_name.split(" + ")
            if layer_name.startswith("Reformatting CopyNode for Input Tensor") or layer_name.startswith("Reformatting CopyNode for Output Tensor"):
                p = layer_name.find(" to ")
                if p != -1:
                    layer_name = layer_name[p+4:]
                    onnx_names = layer_name.split(" + ")
                    for i in range(len(onnx_names)):
                        if onnx_names[i].startswith("reshape_before_"):
                            onnx_names[i] = onnx_names[i][len("reshape_before_"):]
                        elif onnx_names[i].startswith("reshape_after_"):
                            onnx_names[i] = onnx_names[i][len("reshape_after_"):]

        layer["ONNXNames"] = onnx_names
    return layerinfo

def preprocess_profile_json(profile):
    if profile is None:
        return None
    
    for i, item in enumerate(profile):
        item["idd"] = i

    return {item["name"]: item for item in profile if "name" in item}

def make_onnx_view(db:Session, metadata):
    print(f"make_onnx_view: {metadata}")
    model     = metadata["model"]
    layerinfo = metadata.get("layerinfo", None)
    profile   = metadata.get("profile", None)
    buildlog  = metadata.get("buildlog", None)
    trtperf   = metadata.get("trtperf", None)
    layerinfo_compared = metadata.get("layerinfo_compared", None)
    profile_compared   = metadata.get("profile_compared", None)
    buildlog_compared  = metadata.get("buildlog_compared", None)
    trtperf_compared   = metadata.get("trtperf_compared", None)
    with_kernel_match_report = metadata["with_kernel_match_report"]
    ignore_cache = metadata.get("ignore_cache", False)

    if with_kernel_match_report:
        if any([item is None for item in [model, profile, layerinfo, profile_compared, layerinfo_compared]]):
            return failed("Missing the required files [all] if with_kernel_match_report")
    else:
        if model is None:
            return failed("Missing the required file: model.onnx")
    
    additions = {}
    model     = get_file_instance(db, model)
    buildlog = load_normal_file(db, additions, "buildlog", buildlog)
    buildlog_compared = load_normal_file(db, additions, "buildlog_compared", buildlog_compared)
    layerinfo = load_json_file(db, additions, "layerinfo", layerinfo)
    additions["layerinfo"] = preprocess_layerinfo_json(additions.get("layerinfo"))
    profile   = load_json_file(db, additions, "profile", profile)
    additions["profile"]   = preprocess_profile_json(additions.get("profile"))
    trtperf   = load_csv_file(db, additions, "trtperf", trtperf)
    layerinfo_compared = load_json_file(db, additions, "layerinfo_compared", layerinfo_compared)
    additions["layerinfo_compared"] = preprocess_layerinfo_json(additions.get("layerinfo_compared"))
    profile_compared = load_json_file(db, additions, "profile_compared", profile_compared)
    additions["profile_compared"] = preprocess_profile_json(additions.get("profile_compared"))
    trtperf_compared = load_csv_file(db, additions, "trtperf_compared", trtperf_compared)
    if with_kernel_match_report:
        if any([item is None for item in [model, profile, layerinfo, profile_compared, layerinfo_compared]]):
            return failed("Failed to load the required files [all] if with_kernel_match_report")
    else:
        if model is None:
            return failed("Failed to read file: model.onnx")

    folder      = metadata["folder"]
    view_name   = metadata["name"]
    model_path, model_instance  = model
    try:
        model_proto = onnx.load(model_path, load_external_data=False)
    except Exception as e:
        traceback.print_exception(e)
        return failed("Failed to load onnx proto.")

    try:
        model_proto = onnx.shape_inference.infer_shapes(model_proto)
    except Exception as e:
        traceback.print_exception(e)

    if buildlog is not None:
        buildlog_path, buildlog_instance = buildlog
        with open(buildlog_path, "r") as f:
            buildlog_content = f.read()
            start = buildlog_content.find("=== Performance summary ===")
            mid   = buildlog_content.find("Total GPU Compute Time:", start + len("=== Performance summary ===") + 1) if start != -1 else -1
            end   = buildlog_content.find("\n", mid) if mid != -1 else -1
            if end != -1:
                cleanup_lines = []
                for line in buildlog_content[start:end].strip().split("\n"):
                    p = line.find("[I] ")
                    if p != -1:
                        line = line[p+4:]
                        cleanup_lines.append(line)
                metadata["performance_summary"] = "\n".join(cleanup_lines)
    else:
        metadata["performance_summary"] = None
    
    if buildlog_compared is not None:
        buildlog_compared_path, buildlog_compared_instance = buildlog_compared
        with open(buildlog_compared_path, "r") as f:
            buildlog_compared_content = f.read()
            start = buildlog_compared_content.find("=== Performance summary ===")
            mid   = buildlog_compared_content.find("Total GPU Compute Time:", start + len("=== Performance summary ===") + 1) if start != -1 else -1
            end   = buildlog_compared_content.find("\n", mid) if mid != -1 else -1
            if end != -1:
                cleanup_lines = []
                for line in buildlog_compared_content[start:end].strip().split("\n"):
                    p = line.find("[I] ")
                    if p != -1:
                        line = line[p+4:]
                        cleanup_lines.append(line)
                metadata["performance_summary_compared"] = "\n".join(cleanup_lines)
    else:
         metadata["performance_summary_compared"] = None

    total_diff_full_latency = 0
    total_diff_exclude_empty_latency = 0
    if with_kernel_match_report:
        name_mapping = {node.name: node.op_type for node in model_proto.graph.node}
        if not generate_kernel_match_report(name_mapping, "ThorU", additions["layerinfo"], additions["profile"], "OrinX", additions["layerinfo_compared"], additions["profile_compared"], additions):
            return failed("Failed to generate kernel match report.")
        
        groups = additions["kernel_match"]
        for key in groups:
            group = groups[key]
            ThorU = group["ThorU"]
            OrinX = group["OrinX"]

            sum_of_thoru_latency = sum([item["profile"].get("averageMs", 0) for item in ThorU if "profile" in item])
            sum_of_orinx_latency = sum([item["profile"].get("averageMs", 0) for item in OrinX if "profile" in item])
            total_diff_full_latency += max(sum_of_thoru_latency - sum_of_orinx_latency, 0)

            if len(OrinX) > 0:
                total_diff_exclude_empty_latency += max(sum_of_thoru_latency - sum_of_orinx_latency, 0)

    metadata["metrics"] = dict(
        total_diff_full_latency=total_diff_full_latency,
        total_diff_exclude_empty_latency=total_diff_exclude_empty_latency
    )
    additions["view_meta"] = dict(
        virtual_folder = folder,
        name = view_name,
        view_type = "onnx_with_kernel_match" if with_kernel_match_report else "onnx",
        metadata = metadata
    )

    additions["model_meta"] = dict(
        file_id=model_instance.id,
        name=model_instance.name,
        description=model_instance.description,
        size_bytes=model_instance.size_bytes,
        virtual_folder=model_instance.virtual_folder,
        create_time=format_date(model_instance.create_time),
        update_time=format_date(model_instance.update_time),
        with_kernel_match_report=with_kernel_match_report
    )
    
    layout_caching_path = os.path.join(ONNX_LAYOUT_CACHING_ROOT, f"{model_instance.signature}.json")
    if ignore_cache:
        print(f"Ignore cache for: {model_instance.signature}")
        layout_caching_path = None

    with NamedTemporaryFile(delete=False) as tf:
        if not onnx_layout_generate(model_proto, tf.name, additions, layout_caching_path):
            return failed("Failed to run graph layout.")

    metadata_string = json.dumps(metadata)
    instance = dao.create_view(db, schemas.View(
        virtual_folder=folder,
        name=view_name,
        view_type="onnx_with_kernel_match" if with_kernel_match_report else "onnx",
        meta_data=metadata_string
    ))
    view_id = instance.id
    view_folder = os.path.join(DATA_ROOT, "views", f"{view_id:012d}")
    model_graph_path = os.path.join(view_folder, "model.graph.json")
    os.makedirs(view_folder, exist_ok=True)
    shutil.move(tf.name, model_graph_path)
    db.commit()
    return success(view_id=view_id, metrics=metadata["metrics"])

def make_trex_view(db:Session, metadata, related_view=None):
    layerinfo = metadata.get("layerinfo", None)
    profile   = metadata.get("profile", None)
    folder    = metadata.get("folder", None)
    view_name = metadata.get("name", None)
    if layerinfo is None or folder is None or view_name is None:
        return failed("Missing the required file: layerinfo.json, folder, view_Name")
    
    layerinfo = get_file_instance(db, layerinfo)
    profile   = get_file_instance(db, profile)
    if layerinfo is None:
        return failed("Failed to read file: layerinfo.json")
    
    layerinfo_path, layerinfo_instance = layerinfo
    profile_path, profile_instance     = None, None
    if profile is not None:
        profile_path, profile_instance = profile

    with NamedTemporaryFile(delete=False, dir=TEMP_DIR) as tf:
        if not generate_graph_svg(layerinfo_path, profile_path, tf.name):
            return failed("Failed to make svg graph.")

    metadata = json.dumps(metadata)
    instance = dao.create_view(db, schemas.View(
        virtual_folder=folder,
        name=view_name,
        view_type="trex",
        meta_data=metadata
    ))
    view_id = instance.id
    view_folder = os.path.join(DATA_ROOT, "views", f"{view_id:012d}")
    os.makedirs(view_folder, exist_ok=True)
    view_svg_path = os.path.join(view_folder, "view.svg")
    shutil.move(tf.name, view_svg_path)

    if related_view is not None:
        view_meta_data = json.loads(related_view.meta_data)
        view_meta_data["related_trex_view_id"] = view_id
        related_view.meta_data = json.dumps(view_meta_data)

    db.commit()
    if related_view is not None:
        return view_id
    
    return success(view_id=view_id)

@app.post("/find_file_by_signature/{signature}")
async def find_file_by_signature(signature: str, db: Session = default_db_depends):
    file = dao.find_file_by_signature(db, signature)
    if file is None:
        return success(file=None)
    
    return success(
        file=dict(
            file_id=file.id,
            file_name=file.name,
            size_bytes=file.size_bytes,
            create_time=format_date(file.create_time),
            update_time=format_date(file.update_time),
            virtual_folder=file.virtual_folder,
            description=file.description
        )
    )

@app.post("/list_views_by_keyword")
async def list_views_by_keyword(request: Request, db: Session = default_db_depends):
    params = await request.json()
    keywords = params.get("keywords", "")
    page = params.get("page", 0)
    pagesize = params.get("pagesize", 20)
    page = max(0, page)
    pagesize = max(1, min(pagesize, 20))
    views, count = dao.find_view_by_keyword(db, keywords, page, pagesize)
    resp = [
        dict(
            idd = view.id,
            name = view.name,
            virtual_folder = view.virtual_folder,
            view_type = view.view_type,
            meta_data = json.loads(view.meta_data) if view.meta_data else {},
            create_time = format_date(view.create_time),
            update_time = format_date(view.update_time)
        ) for view in views
    ]
    return success(views=resp, count=count)

@app.post("/health_check/{view_id}")
async def health_check(view_id: int, db: Session = default_db_depends):

    health_check_path = os.path.join(DATA_ROOT, "views", f"{view_id:012d}", "health_check.json")
    if os.path.exists(health_check_path):
        old_health_check_time = os.path.getmtime(health_check_path)
        health_check_script_time = os.path.getmtime("health_check.py")
        if old_health_check_time >= health_check_script_time:
            with open(health_check_path, "r") as f:
                return success(issues=json.load(f))

    view_instance = dao.find_view_by_id(db, view_id)
    if view_instance is None:
        return failed(f"Unknow view_id: {view_id}")
    
    if view_instance.view_type not in ["onnx", "onnx_with_kernel_match"]:
        return failed(f"Unknow view type: {view_instance.view_type}.")

    view_meta_data = json.loads(view_instance.meta_data)
    model_id = view_meta_data.get("model", None)
    if model_id is None:
        return failed("Can not found the model in the view meta data.")
    
    model_path = os.path.join(DATA_ROOT, "files", f"{model_id:012d}")
    if not os.path.exists(model_path):
        return failed(f"Can not found the model in the file system: {model_id}")
    
    model_proto = onnx.load(model_path, load_external_data=False)
    checker = HealthChecker(model_proto)
    issues = checker.check_all()

    with open(health_check_path, "w") as f:
        json.dump(issues, f, indent=4)

    return success(issues=issues)

@app.post("/create_view")
async def create_view(request: Request, db: Session = default_db_depends):
    metadata = await request.json()
    view_type = metadata["view_type"]
    if view_type == "onnx":
        return make_onnx_view(db, metadata)
    elif view_type == "trex":
        return make_trex_view(db, metadata)
    else:
        return failed(f"Unknow view type: {view_type}.")

@app.get("/related_trex_view/{view_id}")
async def related_trex_view(view_id: int, db: Session = default_db_depends):
    view_instance = dao.find_view_by_id(db, view_id)
    if view_instance is None:
        return HTMLResponse(f"Unknow view_id: {view_id}")
    
    if view_instance.view_type not in ["onnx", "onnx_with_kernel_match"]:
        return HTMLResponse(f"Unknow view type: {view_instance.view_type}.")
    
    view_meta_data = json.loads(view_instance.meta_data)
    related_view_id = view_meta_data.get("related_trex_view_id", None)
    if related_view_id is not None:
        return RedirectResponse(f"/view/{related_view_id}")

    meta_data = dict(
        layerinfo = view_meta_data.get("layerinfo", None),
        profile = view_meta_data.get("profile", None),
        folder = view_instance.virtual_folder,
        description = f"Related to {view_id}",
        name = view_instance.name,
        view_type = "trex",
        related_view_id = view_id
    )
    view_id = make_trex_view(db, meta_data, view_instance)
    return RedirectResponse(f"/view/{view_id}")

@app.post("/get_view_info/{view_id}")
def get_view_info(view_id: int, db: Session = default_db_depends):
    view_instance = dao.find_view_by_id(db, view_id)
    if view_instance is None:
        return failed(f"Unknow view_id: {view_id}")
    
    def fill_file_info(meta_data, key):
        if key in meta_data:
            file = get_file_instance(db, meta_data[key])
            if file is not None:
                meta_data[key] = dict(file_id=file[1].id, name=file[1].name, description=file[1].description, size_bytes=file[1].size_bytes, virtual_folder=file[1].virtual_folder, create_time=format_date(file[1].create_time), update_time=format_date(file[1].update_time))

    meta_data = json.loads(view_instance.meta_data) if view_instance.meta_data else {}
    if view_instance.view_type == "onnx":
        fill_file_info(meta_data, "model")
        fill_file_info(meta_data, "layerinfo")
        fill_file_info(meta_data, "profile")
        fill_file_info(meta_data, "buildlog")
        fill_file_info(meta_data, "trtperf")
    elif view_instance.view_type == "onnx_with_kernel_match":
        fill_file_info(meta_data, "model")
        fill_file_info(meta_data, "layerinfo")
        fill_file_info(meta_data, "profile")
        fill_file_info(meta_data, "buildlog")
        fill_file_info(meta_data, "trtperf")
        fill_file_info(meta_data, "layerinfo_compared")
        fill_file_info(meta_data, "profile_compared")
        fill_file_info(meta_data, "buildlog_compared")
        fill_file_info(meta_data, "trtperf_compared")
    elif view_instance.view_type == "trex":
        fill_file_info(meta_data, "layerinfo")
        fill_file_info(meta_data, "profile")

    return success(
        idd = view_instance.id,
        name = view_instance.name,
        virtual_folder = view_instance.virtual_folder,
        view_type = view_instance.view_type,
        meta_data = meta_data,
        create_time = format_date(view_instance.create_time),
        update_time = format_date(view_instance.update_time)
    )

@app.post("/get_view_metadata/{view_id}")
def get_view_metadata(view_id: int, db: Session = default_db_depends):
    view_instance = dao.find_view_by_id(db, view_id)
    if view_instance is None:
        return failed(f"Unknow view_id: {view_id}")
    
    meta_data = json.loads(view_instance.meta_data) if view_instance.meta_data else {}
    return success(**meta_data)

@app.post("/generate_profile_layers_from_logfile")
async def generate_profile_layers_from_logfile(request: Request, db: Session = default_db_depends):
    params = await request.json()
    build_log_file_id = params.get("build_log_file_id", None)
    if build_log_file_id is None:
        return failed("Missing the required file: build_log_file_id")
    
    build_log_file = get_file_instance(db, build_log_file_id)
    if build_log_file is None:
        return failed("Failed to read file: build_log_file_id")
    
    build_log_path, build_log_instance = build_log_file
    projection_code = params.get("projection_code", None)
    if projection_code is not None and projection_code.find("import ") != -1:
        return failed("Invalid projection code: " + projection_code)

    try:
        layers, profile, onnx_path = parse_layer_info_and_performance_and_model(build_log_path, projection_code)
    except Exception as e:
        traceback.print_exc()
        return failed("Failed to parse the build log file: " + str(e))
    
    if onnx_path is not None and not os.path.exists(onnx_path):
        return failed("Can not found the onnx model from path: " + onnx_path)
    
    layers_file_path = None
    profile_file_path = None
    if layers is not None:
        with NamedTemporaryFile(delete=False, dir=TEMP_DIR) as layers_file:
            with open(layers_file.name, "w") as f:
                json.dump(layers, f, indent=4)
            layers_file_path = layers_file.name
            layers_signature = signature_file(layers_file_path)
        
    if profile is not None:
        with NamedTemporaryFile(delete=False, dir=TEMP_DIR) as profile_file:
            with open(profile_file.name, "w") as f:
                json.dump(profile, f, indent=4)
            profile_file_path = profile_file.name
            profile_signature = signature_file(profile_file_path)

    storage_folder  = os.path.join(DATA_ROOT, "files")
    os.makedirs(storage_folder, exist_ok=True)

    base_name = os.path.splitext(build_log_instance.name)[0]
    output = {"layers": None, "profile": None}
    if layers_file_path is not None:
        instance = dao.create_file(db, schemas.File(
            virtual_folder=build_log_instance.virtual_folder,
            name=base_name + ".layers.json",
            signature=layers_signature,
            description=build_log_instance.description,
            size_bytes=os.path.getsize(layers_file_path)
        ))
        file_id = instance.id
        path = os.path.join(storage_folder, f"{file_id:012d}")
        shutil.move(layers_file_path, path)
        output["layers"] = dict(
            file_id=instance.id,
            file_name=instance.name,
            size_bytes=instance.size_bytes,
            create_time=format_date(instance.create_time),
            update_time=format_date(instance.update_time),
            virtual_folder=instance.virtual_folder,
            description=instance.description
        )

    if profile_file_path is not None:
        instance = dao.create_file(db, schemas.File(
            virtual_folder=build_log_instance.virtual_folder,
            name=base_name + ".profile.json",
            signature=profile_signature,
            description=build_log_instance.description,
            size_bytes=os.path.getsize(profile_file_path)
        ))
        file_id = instance.id
        path = os.path.join(storage_folder, f"{file_id:012d}")
        shutil.move(profile_file_path, path)
        output["profile"] = dict(
            file_id=instance.id,
            file_name=instance.name,
            size_bytes=instance.size_bytes,
            create_time=format_date(instance.create_time),
            update_time=format_date(instance.update_time),
            virtual_folder=instance.virtual_folder,
            description=instance.description
        )

    if onnx_path is not None:
        instance = dao.create_file(db, schemas.File(
            virtual_folder=build_log_instance.virtual_folder,
            name=base_name + ".onnx",
            signature=signature_file(onnx_path),
            description=build_log_instance.description,
            size_bytes=os.path.getsize(onnx_path)
        ))
        file_id = instance.id
        path = os.path.join(storage_folder, f"{file_id:012d}")
        if os.path.exists(path):
            os.remove(path)
            
        os.symlink(onnx_path, path)
        output["onnx"] = dict(
            file_id=instance.id,
            file_name=instance.name,
            size_bytes=instance.size_bytes,
            create_time=format_date(instance.create_time),
            update_time=format_date(instance.update_time),
            virtual_folder=instance.virtual_folder,
            description=instance.description
        )
    db.commit()
    return success(**output)

@app.get("/view/{view_id}")
def show_view(view_id: str, db: Session = default_db_depends):
    try:
        view_id = int(view_id)
    except Exception as e:
        return HTMLResponse(f"Unknow view_id: {view_id}")
    
    view_instance = dao.find_view_by_id(db, view_id)
    if view_instance is None:
        return HTMLResponse(f"Unknow view_id: {view_id}")

    view_type = view_instance.view_type
    view_name = view_instance.name.replace("\"", "") if view_instance.name is not None else ""
    if view_type in ["onnx", "onnx_with_kernel_match"]:
        with open("static/onnx.html", "r") as f:
            content = f.read()
            graph_json_path = os.path.join("/datas", "views", f"{view_id:012d}", "model.graph.json")
            content = content.replace('<meta name="onnx-json-url" content=""/>', f'<meta name="onnx-json-url" content="{graph_json_path}" />\n\t<meta name="view-id" content="{view_id}" />\n\t<meta name="view-name" content="{view_name}" />')
        return HTMLResponse(content)
    elif view_type == "trex":
        with open("static/trex.html", "r") as f:
            content = f.read()
            view_svg_path = os.path.join("/datas", "views", f"{view_id:012d}", "view.svg")
            content = content.replace('<meta name="view-svg-path" content=""/>', f'<meta name="view-svg-path" content="{view_svg_path}" />\n\t<meta name="view-id" content="{view_id}" />\n\t<meta name="view-name" content="{view_name}" />')
        return HTMLResponse(content)
    
def download_file(url, saveto, chunk_size=4096*10):
    try:
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.104 Safari/537.36 Core/1.53.2372.400 QQBrowser/9.5.10548.400'
        }
        response = requests.get(url, stream=True, verify=False, headers=headers)
        if response.status_code != 200:
            msg = str(response.content, encoding="utf-8")
            print(f"Download failed, {msg}")
            return None

        total_file_size = 0
        signaturer   = hashlib.md5()
        content_iter = response.iter_content(chunk_size=chunk_size)
        with open(saveto, "wb") as f:
            for chunk in content_iter:
                total_file_size += len(chunk)
                signaturer.update(chunk)
                f.write(chunk)

        return signaturer.hexdigest() + f"{total_file_size:016d}", total_file_size
    except Exception as e:
        traceback.print_exc()
    return None
    
def download_file_from_nvbug(url, saveto, token, chunk_size=4096*10):
    try:
        attachment_id = url.split("attachmentguid=")[1]
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.104 Safari/537.36 Core/1.53.2372.400 QQBrowser/9.5.10548.400',
            'Authorization': f'Bearer {token}'
        }
        real_url = f"https://prod.api.nvidia.com/int/nvbugs/api/Attachment/DownloadAttachment?attachmentguid={attachment_id}"
        response = requests.get(real_url, stream=True, verify=False, headers=headers)
        if response.status_code != 200:
            msg = str(response.content, encoding="utf-8")
            print(f"Download failed, {msg}")
            return None

        total_file_size = 0
        signaturer   = hashlib.md5()
        content_iter = response.iter_content(chunk_size=chunk_size)
        with open(saveto, "wb") as f:
            for chunk in content_iter:
                total_file_size += len(chunk)
                signaturer.update(chunk)
                f.write(chunk)

        return signaturer.hexdigest() + f"{total_file_size:016d}", total_file_size
    except Exception as e:
        traceback.print_exc()
    return None

def download_file_from_ssh(path, ip, password, saveto, chunk_size=4096*10):
    try:
        print(f"sshpass -p \"{password}\" scp -o StrictHostKeyChecking=no {ip}:{path} {saveto}")
        code = os.system(f"sshpass -p \"{password}\" scp -o StrictHostKeyChecking=no {ip}:{path} {saveto}")
        if code != 0:
            print(f"Failed to download file from ssh: {path}")
            return None

        total_file_size = 0
        signaturer      = hashlib.md5()
        with open(saveto, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if len(data) == 0:
                    break

                total_file_size += len(data)
                signaturer.update(data)

        return signaturer.hexdigest() + f"{total_file_size:016d}", total_file_size
    except Exception as e:
        traceback.print_exc()
    return None

def extract_file_name_by_url(url):
    if url is None: return None
    p = url.find("?")
    q = url.find("#")
    i = min(p, q)
    if i != -1:
        url = url[:i]
    
    e = url.rfind("/")
    if e != -1:
        url = url[e + 1:]
    return url

def extract_file_name_by_ssh(url):
    if url is None: return None
    i = url.rfind("/")
    if i != -1:
        url = url[i + 1:]
    return url

@app.get("/download/{file_id}")
def download(file_id: int, db: Session = default_db_depends):
    file = get_file_instance(db, file_id)
    if file is None:
        # print(f"Failed to get_file_instance for: {file_id}")
        return HTMLResponse(f"Unknow file: {file_id}", 404)
    
    path, instance = file
    return FileResponse(path, 200, filename=instance.name)


@app.post("/create_file_from_url")
async def create_file_from_url(request: Request, db: Session = default_db_depends):
    params = await request.json()
    if "file_url" not in params:
        return failed("Unknow file_url")
    
    file_url = params["file_url"]
    file_name = params.get("file_name", None)
    if file_name is None or file_name == "":
        file_name = extract_file_name_by_url(file_url)

    folder      = params.get("folder", "default")
    description = params.get("description", f"From URL: {file_url}")

    with NamedTemporaryFile(delete=False, dir=TEMP_DIR) as tf:
        signature = download_file(file_url, tf.name)
        if signature is None:
            return failed(f"Download failed from url: {file_url}")
    
    instance = dao.create_file(db, schemas.File(
        virtual_folder=folder,
        name=file_name,
        signature=signature[0],
        description=description,
        size_bytes=signature[1]
    ))
    file_id = instance.id
    folder  = os.path.join(DATA_ROOT, "files")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{file_id:012d}")
    shutil.move(tf.name, path)
    db.commit()
    return success(file_id=file_id, file_name=file_name)


@app.post("/create_file_from_nvbug")
async def create_file_from_nvbug(request: Request, db: Session = default_db_depends):
    params = await request.json()
    if "nvbug_url" not in params or "token" not in params or "file_name" not in params or "description" not in params or "folder" not in params:
        return failed("Unknow nvbug_url.")
    
    nvbug_url = params["nvbug_url"]
    token     = params["token"]
    file_name = params["file_name"]
    if file_name is None or file_name == "":
        file_name = extract_file_name_by_url(nvbug_url)

    folder      = params.get("folder", "default")
    description = params.get("description", f"From URL: {nvbug_url}")

    with NamedTemporaryFile(delete=False, dir=TEMP_DIR) as tf:
        signature = download_file_from_nvbug(nvbug_url, tf.name, token)
        if signature is None:
            return failed(f"Download failed from NVBug: {nvbug_url}")
    
    instance = dao.create_file(db, schemas.File(
        virtual_folder=folder,
        name=file_name,
        signature=signature[0],
        description=description,
        size_bytes=signature[1]
    ))
    file_id = instance.id
    folder  = os.path.join(DATA_ROOT, "files")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{file_id:012d}")
    shutil.move(tf.name, path)
    db.commit()
    return success(file_id=file_id, file_name=file_name)

@app.post("/create_file_from_ssh")
async def create_file_from_ssh(request: Request, db: Session = default_db_depends):
    params = await request.json()
    if "ip" not in params or "path" not in params or "password" not in params or "file_name" not in params or "description" not in params or "folder" not in params:
        return failed("Unknow nvbug_url.")
    
    ip        = params["ip"]
    sshpath   = params["path"]
    password  = params["password"]
    file_name = params["file_name"]
    if file_name is None or file_name == "":
        file_name = extract_file_name_by_ssh(sshpath)

    folder      = params.get("folder", "default")
    description = params.get("description", f"From SSH: {sshpath}")

    with NamedTemporaryFile(delete=False, dir=TEMP_DIR) as tf:
        signature = download_file_from_ssh(sshpath, ip, password, tf.name)
        if signature is None:
            return failed(f"Download failed from SSH: {sshpath}")
    
    instance = dao.create_file(db, schemas.File(
        virtual_folder=folder,
        name=file_name,
        signature=signature[0],
        description=description,
        size_bytes=signature[1]
    ))
    file_id = instance.id
    folder  = os.path.join(DATA_ROOT, "files")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{file_id:012d}")
    shutil.move(tf.name, path)
    db.commit()
    return success(file_id=file_id, file_name=file_name)


@app.post("/get_info_from_file_id/{file_id}")
async def get_info_from_file_id(file_id: int, db: Session = default_db_depends):
    
    file = get_file_instance(db, file_id)
    if file is None:
        # print(f"Failed to get_file_instance for: {file_id}")
        return failed(f"Unknow file: {file_id}")
    
    path, instance = file
    return success(file_name=instance.name, file_id=instance.id, description=instance.description, virtual_folder=instance.virtual_folder, create_time=format_date(instance.create_time), update_time=format_date(instance.update_time), size_bytes=instance.size_bytes)
    
@app.post("/create_new_issue")
async def create_new_issue(request: Request, db: Session = default_db_depends):
    params = await request.json()
    if "view_id" not in params or "keywords" not in params or "description" not in params or "creator" not in params or "associate_nodes" not in params:
        return failed("Missing param in the request.")
    
    with db.begin():
        issue = dao.create_issue(db, schemas.Issue(
            keywords=params["keywords"],
            description=params["description"],
            view_id=params["view_id"],
            creator=params["creator"],
            associate_nodes=json.dumps(params["associate_nodes"])
        ))
        return success(issue_id=issue.id, keywords=issue.keywords, description=issue.description, creator=issue.creator, create_time=format_date(issue.create_time), update_time=format_date(issue.update_time), associate_nodes=json.loads(issue.associate_nodes))

@app.post("/get_issuelist_by_view_id/{view_id}")
async def get_issuelist_by_view_id(view_id: int, db: Session = default_db_depends):
    with db.begin():
        issues = dao.find_issues_by_view_id(db, view_id, "normal")
        return success(issues=[
            dict(
                issue_id=issue.id, keywords=issue.keywords, description=issue.description, creator=issue.creator, create_time=format_date(issue.create_time), update_time=format_date(issue.update_time), associate_nodes=json.loads(issue.associate_nodes)
            )
            for issue in issues
        ])

@app.post("/delete_issue/{issue_id}")
async def delete_issue(issue_id: int, db: Session = default_db_depends):
    with db.begin():
        issue = dao.find_issue_by_id(db, issue_id, "normal")
        if issue is None:
            return failed(f"Not found issue: {issue_id}")
        issue.status = "delete"
        return success()

@app.post("/update_issue")
async def update_issue(request: Request, db: Session = default_db_depends):
    params = await request.json()
    if "issue_id" not in params or "keywords" not in params or "description" not in params or "creator" not in params or "associate_nodes" not in params:
        return failed("Missing param in the request.")
    
    with db.begin():
        issue_id = params["issue_id"]
        issue = dao.find_issue_by_id(db, issue_id, "normal")
        if issue is None:
            return failed(f"Not found issue: {issue_id}")
        
        issue.keywords = params["keywords"]
        issue.description = params["description"]
        issue.creator = params["creator"]
        issue.associate_nodes = json.dumps(params["associate_nodes"])
        issue.update_time = datetime.now()
        return success(issue_id=issue.id, keywords=issue.keywords, description=issue.description, creator=issue.creator, create_time=format_date(issue.create_time), update_time=format_date(issue.update_time), associate_nodes=json.loads(issue.associate_nodes))

@app.post("/collect_samples")
async def collect_samples(request: Request):
    params = await request.json()
    samples = params["samples"]
    current_sample_id = int(open("current_sample_id", "r").read())
    for sample in samples:
        with open(f"samples/sample_{current_sample_id:05d}.txt", "w") as f:
            f.write(sample)
        current_sample_id += 1

    open("current_sample_id", "w").write(str(current_sample_id))
    return success()

@app.get("/subgraph/{view_id}")
def home(view_id:int, layerids:str, db: Session = default_db_depends):
    if layerids is None or layerids == "" or not re.fullmatch("[\\d,]+", layerids):
        return HTMLResponse(f"<html></body>Invalid layerids: {layerids}</body></html>", 403)

    view_instance = dao.find_view_by_id(db, view_id)
    if view_instance is None:
        return HTMLResponse(f"Unknow view_id: {view_id}", 404)
    
    meta_data = json.loads(view_instance.meta_data)
    model_file_id = meta_data.get("model", None)
    if model_file_id is None:
        return HTMLResponse(f"Unknow view_id: {view_id}", 404)
    
    model_file = get_file_instance(db, model_file_id)
    if model_file is None:
        # print(f"Failed to get_file_instance for: {model_file_id}")
        return HTMLResponse(f"Unknow view_id: {view_id}", 404)
    
    model_path, model_instance = model_file
    layerids = set(map(int, layerids.split(",")))
    if len(layerids) == 0:
        print("Empty layerids.")
        return HTMLResponse(f"UEmpty layerids. view_id = {view_id}", 404)

    try:
        model = onnx.load(model_path, load_external_data=False)
    except Exception as e:
        print(f"Failed to load file: {model_path}")
        return HTMLResponse(f"Unknow view_id: {view_id}", 500)

    if model is None:
        print(f"Failed to load file: {model_path}")
        return HTMLResponse(f"Unknow view_id: {view_id}", 404)
    
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        traceback.print_exception(e)

    initializer_mapping = set([item.name for item in model.graph.initializer])
    constant_output_to_nodeid_mapping = {node.output[0]:node_id + 1 for node_id, node in enumerate(model.graph.node) if node.op_type == "Constant"}
    reference_initializers = set()
    reference_constant_node_id = set()
    reference_constant_inp       = set()
    for i, node in enumerate(model.graph.node):
        node_id = i + 1
        if node_id in layerids:
            for inp in node.input:
                if inp in initializer_mapping:
                    reference_initializers.add(inp)
                elif inp in constant_output_to_nodeid_mapping:
                    reference_constant_node_id.add(constant_output_to_nodeid_mapping[inp])
                    reference_constant_inp.add(inp)
            
    for i in range(len(model.graph.node)-1, -1, -1):
        node_id = i + 1
        if node_id not in layerids and node_id not in reference_constant_node_id:
            del model.graph.node[i]

    for i in range(len(model.graph.initializer)-1, -1, -1):
        if model.graph.initializer[i].name not in reference_initializers:
            del model.graph.initializer[i]

    input_to_node  = {}
    output_to_node = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            continue

        for inp in node.input:
            if inp != "":
                input_to_node[inp] = node

        for out in node.output:
            output_to_node[out] = node

    network_outputs   = {item.name: item for item in model.graph.output}
    undefined_inputs  = []
    undefined_outputs = []
    for inp in input_to_node:
        if inp not in output_to_node and inp not in reference_initializers and inp not in reference_constant_inp:
            undefined_inputs.append(inp)

    for out in output_to_node:
        if out in input_to_node and out in network_outputs or out not in input_to_node:
            undefined_outputs.append(out)

    tensor_info_mapping = {item.name : item for item in model.graph.value_info}
    for inp in model.graph.input:
        tensor_info_mapping[inp.name] = inp

    for out in model.graph.output:
        tensor_info_mapping[out.name] = out

    for i in range(len(model.graph.input)-1, -1, -1):
        if model.graph.input[i].name not in undefined_inputs:
            del model.graph.input[i]
        else:
            del undefined_inputs[undefined_inputs.index(model.graph.input[i].name)]
    
    for i in range(len(model.graph.output)-1, -1, -1):
        if model.graph.output[i].name not in undefined_outputs:
            del model.graph.output[i]
        else:
            del undefined_outputs[undefined_outputs.index(model.graph.output[i].name)]

    for inp in undefined_inputs:
        if inp in tensor_info_mapping:
            info = tensor_info_mapping[inp]
            model.graph.input.append(onnx.ValueInfoProto(name=inp, type=info.type))
        else:
            model.graph.input.append(onnx.ValueInfoProto(name=inp))
    
    for out in undefined_outputs:
        if out in tensor_info_mapping:
            info = tensor_info_mapping[out]
            model.graph.output.append(onnx.ValueInfoProto(name=out, type=info.type))
        else:
            model.graph.output.append(onnx.ValueInfoProto(name=out))

    bio = BytesIO()
    onnx.save_model(model, bio)
    filesize = bio.tell()
    bio.seek(0, 0)
    del model

    file_name = os.path.splitext(model_instance.name)[0] + "-sg" + str(len(layerids)) + ".onnx"
    return StreamingResponse(bio, headers={"Content-Length": str(filesize), 'Content-Disposition': f'attachment; filename="{file_name}"'})

@app.post("/execute_graph_code/{view_id}")
async def execute_graph_code(view_id:int, request:Request, db: Session = default_db_depends):
    params = await request.json()
    code = params["code"]
    load_onnx = params["load_onnx"]
    name = params["name"].strip()
    if name.find("..") != -1 or name.find("/") != -1 or name.find("\\") != -1:
        return failed("Invalid name: " + name)
    
    model_proto = None
    if load_onnx:
        view_instance = dao.find_view_by_id(db, view_id)
        if view_instance is None:
            return failed(f"Unknow view_id: {view_id}")
        
        if view_instance.view_type not in ["onnx", "onnx_with_kernel_match"]:
            return failed(f"Unknow view type: {view_instance.view_type}.")

        view_meta_data = json.loads(view_instance.meta_data)
        model_id = view_meta_data.get("model", None)
        if model_id is None:
            return failed("Can not found the model in the view meta data.")
        
        model_path = os.path.join(DATA_ROOT, "files", f"{model_id:012d}")
        if not os.path.exists(model_path):
            return failed(f"Can not found the model in the file system: {model_id}")
        
        model_proto = onnx.load(model_path, load_external_data=False)

    output = generate_onnx_from_code(code, model_proto)
    if output["status"] == "success":
        meta_file = os.path.join(DATA_ROOT, "views", f"{view_id:012d}", "subgraph_meta.json")
        subgraph_meta = {}
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                subgraph_meta = json.load(f)

        if name not in subgraph_meta:
            subgraph_meta[name] = dict(
                code = code,
                create_time = format_date(datetime.now()),
                profiling_files = [],
                update_time = None
            )
        else:
            subgraph_meta[name]["code"] = code
            subgraph_meta[name]["update_time"] = format_date(datetime.now())

        with open(meta_file, "w") as f:
            json.dump(subgraph_meta, f)

        output_folder = os.path.join(DATA_ROOT, "views", f"{view_id:012d}", "profiling", name)
        model_file  = os.path.join(output_folder, "model.onnx")
        os.makedirs(output_folder, exist_ok=True)
        onnx.save(output["model"], model_file)
        output["model"] = dump_onnx(output["model"])
        return success(
            running_status="success",
            model=output["model"],
            console_output=output["console_output"],
            code_metas=output["code_metas"]
        )
    return success(
        running_status="error", 
        message=output["message"], 
        traceback=output["traceback"], 
        console_output=output["console_output"]
    )

@app.get("/get_logfile_data/{file_id}/{type}")
async def get_logfile_data(file_id:int, type:str, db: Session = default_db_depends):

    logfile = get_file_instance(db, file_id)
    if logfile is None:
        return failed(f"Can not found the logfile: {file_id}")
    
    logfile_path, logfile_instance = logfile
    if not os.path.exists(logfile_path):
        return failed(f"Can not found the file: {file_id}")

    with open(logfile_path, "r") as f:
        content = f.read()

    if type == "command":
        cmds = re.findall("&&&& PASSED (.*)|&&&& RUNNING (.*)", content)
        if len(cmds) == 0:
            return failed(f"Can not found the command in file: {file_id}")
        
        cmd = cmds[0]
        cmd = cmd[0] if cmd[0] != "" else cmd[1]
        p = cmd.find("# ")
        if p != -1:
            cmd = cmd[p+2:]
        return success(cmd=cmd)
    return failed(f"Unknow type: {type}")

@app.post("/get_subgraph_metas/{view_id}")
async def get_subgraph_metas(view_id:int, request:Request, db: Session = default_db_depends):
    meta_file = os.path.join(DATA_ROOT, "views", f"{view_id:012d}", "subgraph_meta.json")
    subgraph_meta = {}
    if not os.path.exists(meta_file):
        return success(meta=subgraph_meta)
    
    with open(meta_file, "r") as f:
        subgraph_meta = json.load(f)
    return success(meta=subgraph_meta)

@app.post("/upload_subgraph_code/{view_id}")
async def upload_subgraph_code(view_id:int, request:Request, db: Session = default_db_depends):
    params = await request.json()
    code = params["code"]
    name = params["name"].strip()
    if name.find("..") != -1 or name.find("/") != -1 or name.find("\\") != -1:
        return failed("Invalid name: " + name)
    
    meta_file = os.path.join(DATA_ROOT, "views", f"{view_id:012d}", "subgraph_meta.json")
    subgraph_meta = {}
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            subgraph_meta = json.load(f)

    if name not in subgraph_meta:
        subgraph_meta[name] = dict(
            code = code,
            create_time = format_date(datetime.now()),
            profiling_files = [],
            profiling_time = None
        )
    else:
        subgraph_meta[name]["code"] = code
        subgraph_meta[name]["update_time"] = format_date(datetime.now())

    with open(meta_file, "w") as f:
        json.dump(subgraph_meta, f)

    return success()

@app.post("/delete_subgraph_code/{view_id}")
async def delete_subgraph_code(view_id:int, request:Request, db: Session = default_db_depends):
    params = await request.json()
    name = params["name"].strip()
    if name.find("..") != -1 or name.find("/") != -1 or name.find("\\") != -1 or name == "":
        return failed("Invalid name: " + name)
    
    meta_file = os.path.join(DATA_ROOT, "views", f"{view_id:012d}", "subgraph_meta.json")
    subgraph_meta = {}
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            subgraph_meta = json.load(f)

    if name in subgraph_meta:
        del subgraph_meta[name]
        
    with open(meta_file, "w") as f:
        json.dump(subgraph_meta, f)

    view_folder = os.path.join(DATA_ROOT, "views", f"{view_id:012d}")
    profiling_folder = os.path.join(view_folder, "profiling", name)
    if os.path.exists(profiling_folder):
        shutil.rmtree(profiling_folder)

    return success()

@app.post("/get_profiling_pipeline_status/{view_id}")
async def run_profiling_pipeline(view_id:int, request:Request, db: Session = default_db_depends):
    params = await request.json()
    name = params["name"].strip()
    if name.find("..") != -1 or name.find("/") != -1 or name.find("\\") != -1 or name == "":
        return failed("Invalid name: " + name)
    
    view_folder = os.path.join(DATA_ROOT, "views", f"{view_id:012d}")
    status_file = os.path.join(view_folder, "profiling", name, "status.json")
    if not os.path.exists(status_file):
        return failed(f"No profiling pipeline is started.")

    with open(status_file, "r") as f:
        for i in range(5):
            try:
                status = json.load(f)
                return success(**status)
            except Exception as e:
                time.sleep(0.5)
        
    return failed("Can not read status file.")

@app.post("/run_profiling_pipeline/{view_id}")
async def run_profiling_pipeline(view_id:int, request:Request, db: Session = default_db_depends):
    params = await request.json()
    code = params["code"]
    name = params["name"].strip()
    profiling_script = params["profiling_script"].strip()
    if name.find("..") != -1 or name.find("/") != -1 or name.find("\\") != -1 or name == "":
        return failed("Invalid name: " + name)
    
    view_instance = dao.find_view_by_id(db, view_id)
    if view_instance is None:
        return failed(f"Unknow view_id: {view_id}")
    
    if view_instance.view_type not in ["onnx", "onnx_with_kernel_match"]:
        return failed(f"Unknow view type: {view_instance.view_type}.")
    
    view_meta_data = json.loads(view_instance.meta_data)
    model_id = view_meta_data.get("model", None)
    if model_id is None:
        return failed("Can not found the model in the view meta data.")
    
    model_file = get_file_instance(db, model_id)
    if model_file is None:
        return failed(f"Unknow view_id: {view_id}")

    view_folder = os.path.join(DATA_ROOT, "views", f"{view_id:012d}")
    profiling_folder = os.path.join(view_folder, "profiling", name)
    profiling_outputs = os.path.join(profiling_folder, "outputs")
    if os.path.exists(profiling_outputs):
        shutil.rmtree(profiling_outputs)

    os.makedirs(profiling_folder, exist_ok=True)
    os.makedirs(os.path.join(profiling_folder, "outputs"), exist_ok=True)
    status_file = os.path.join(profiling_folder, "status.json")
    with open(status_file, "w") as f:
        json.dump({"status": "running", "message": "Startup..."}, f)

    model_path, model_instance = model_file
    async_pipeline_runner.commit(
        view_id = view_id,
        onnx_path = model_path,
        code = code,
        profile_name = name,
        status_file = status_file,
        view_folder = view_folder,
        profiling_folder = profiling_folder,
        profiling_script = profiling_script
    )
    return success()

@app.get("/get_profiling_file/{view_id}/{name}/{file}")
async def download_subgraph_by_name(view_id:int, name:str, file:str, db: Session = default_db_depends):
    if name.find("..") != -1 or name.find("/") != -1 or name.find("\\") != -1 or name == "":
        return HTMLResponse("Invalid name: " + name, 403)
    
    if file.find("..") != -1 or file.find("/") != -1 or file.find("\\") != -1 or file == "":
        return HTMLResponse("Invalid file name: " + file, 403)
    
    view_folder = os.path.join(DATA_ROOT, "views", f"{view_id:012d}")
    if file == "ONNX View":
        profiling_file = os.path.join(view_folder, "profiling", name, "model.onnx")
        file = f"{name}_model.onnx"
    else:
        profiling_file = os.path.join(view_folder, "profiling", name, "outputs", file)
        file = f"{name}_{file}"

    if not os.path.exists(profiling_file):
        return HTMLResponse(f"Can not found file for view_id: {view_id}, name: {name}", 404)
    
    return FileResponse(profiling_file, 200, filename=file)

@app.post("/upload_profiling_file/{view_id}")
async def upload_profiling_file(view_id:int, request: Request, file: UploadFile, db: Session = default_db_depends):
    if file is None:
        return failed("unknow item file")
    
    file_name = urlparse.unquote(request.headers.get("file_name", file.filename))
    profile_name = urlparse.unquote(request.headers.get("profile_name", "")).strip()
    if profile_name.find("..") != -1 or profile_name.find("/") != -1 or profile_name.find("\\") != -1 or profile_name == "":
        return failed("Invalid profile_name: " + profile_name)
    
    if file_name.find("..") != -1 or file_name.find("/") != -1 or file_name.find("\\") != -1 or file_name == "":
        return failed("Invalid profile_name: " + file_name)
    
    view_folder = os.path.join(DATA_ROOT, "views", f"{view_id:012d}")
    profiling_folder = os.path.join(view_folder, "profiling", profile_name)
    if not os.path.exists(profiling_folder):
        return failed(f"Can not found profiling by view_id: {view_id} and profile_name: {profile_name}")
    
    meta_file = os.path.join(DATA_ROOT, "views", f"{view_id:012d}", "subgraph_meta.json")
    if not os.path.exists(meta_file):
        return failed(f"Can not found profiling meta by name: {profile_name}")

    with open(meta_file, "r") as f:
        subgraph_meta = json.load(f)

    if profile_name not in subgraph_meta:
        return failed(f"Can not found profiling meta by name: {profile_name}")

    storage_folder = os.path.join(profiling_folder, "inputs")
    os.makedirs(storage_folder, exist_ok=True)

    total_file_size = 0
    CHUNK_SIZE = 1024 * 16
    output_file = os.path.join(storage_folder, file_name)
    async with aiofiles.open(output_file, 'wb') as f:
        while chunk := await file.read(CHUNK_SIZE):
            total_file_size += len(chunk)
            await f.write(chunk)

    new_attached_file = dict(
        name=file_name, size=total_file_size, create_time=format_date(datetime.now())
    )
    profile = subgraph_meta[profile_name]
    attached_files = profile.get("attached_files", [])
    has_same_name = False
    for i, file in enumerate(attached_files):
        if file["name"] == file_name:
            attached_files[i] = new_attached_file
            has_same_name = True
            break

    if not has_same_name:
        attached_files.append(new_attached_file)

    profile["attached_files"] = attached_files
    
    with open(meta_file, "w") as f:
        json.dump(subgraph_meta, f)
    return success(files=attached_files)

@app.post("/remove_profiling_file/{view_id}")
async def run_profiling_pipeline(view_id:int, request:Request, db: Session = default_db_depends):
    params = await request.json()
    profile_name = params["profile_name"].strip()
    file_name = params["file_name"].strip()
    if profile_name.find("..") != -1 or profile_name.find("/") != -1 or profile_name.find("\\") != -1 or profile_name == "":
        return failed("Invalid profile_name: " + profile_name)
    
    if file_name.find("..") != -1 or file_name.find("/") != -1 or file_name.find("\\") != -1 or file_name == "":
        return failed("Invalid file_name: " + file_name)
    
    view_folder = os.path.join(DATA_ROOT, "views", f"{view_id:012d}")
    profiling_folder = os.path.join(view_folder, "profiling", profile_name)
    meta_file = os.path.join(DATA_ROOT, "views", f"{view_id:012d}", "subgraph_meta.json")
    if not os.path.exists(meta_file):
        return failed(f"Can not found profiling meta by name: {profile_name}")

    with open(meta_file, "r") as f:
        subgraph_meta = json.load(f)

    if profile_name not in subgraph_meta:
        return failed(f"Can not found profiling meta by name: {profile_name}")

    profile = subgraph_meta[profile_name]
    attached_files = profile.get("attached_files", [])
    for i, file in enumerate(attached_files):
        if file["name"] == file_name:
            del attached_files[i]
            break

    attach_file_path = os.path.join(profiling_folder, "inputs", file_name)
    if os.path.exists(attach_file_path):
        os.remove(attach_file_path)

    with open(meta_file, "w") as f:
        json.dump(subgraph_meta, f)
    return success(files=attached_files)

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser("Run onnx view server")
    parser.add_argument("--port", type=int, default=8891, help="Run on debug mode")
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)