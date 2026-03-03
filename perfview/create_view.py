import requests
import hashlib
import json
import os
import argparse

# SERVER_URL = "http://10.19.225.242:8891"
SERVER_URL = "http://0.0.0.0:8822"

def signature_file(file_path):
    signaturer = hashlib.md5()
    file_size  = 0
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 64)
            if len(chunk) == 0:
                break
            
            file_size += len(chunk)
            signaturer.update(chunk)

    return signaturer.hexdigest() + f"{file_size:016d}"

def parse_response(response, url):
    string_response = str(response.content, encoding="utf-8")
    if response.status_code != 200:
        print(f"Failed to upload file to {url}, {string_response}")
        return None

    data = json.loads(string_response)
    if data.get("status", "failed") == "ok":
        return data["data"]
    
    print(f"Failed to request {url}, error: {data.get('message', 'unknow error.')}")
    return None

def upload_file(url, local_file, headers, chunk_size=4096*100):
    with open(local_file, "rb") as fhandle:
        def streaming_file():
            while True:
                chunk = fhandle.read(chunk_size)
                if len(chunk) == 0:
                    break
                yield chunk

        response = requests.post(url, data=streaming_file(), headers=headers)
        return parse_response(response, url)

def request_info(url, data=None):
    return parse_response(requests.post(url, json=data), url)

def upload_file_if_not_exist(file_uri, remote_virtual_folder="default", description=""):

    if file_uri is None:
        return dict(file_id = None)

    if file_uri.startswith("http"):
        response = request_info(f"{SERVER_URL}/create_file_from_url", dict(
            file_url = file_uri,
            folder = remote_virtual_folder,
            description = description
        ))
        assert response is not None, f"Failed to create file from url: {file_uri}"
        return response

    if file_uri.startswith("local:"):
        response = request_info(f"{SERVER_URL}/create_file_from_local_path", dict(
            path = file_uri[len("local:"):]
        ))
        assert response is not None, f"Failed to create file from local path: {file_uri}"
        return response["file"]
    
    assert os.path.exists(file_uri), f"File not found: {file_uri}"
    signature   = signature_file(file_uri)
    response = request_info(f"{SERVER_URL}/find_file_by_signature/{signature}")
    assert response is not None, f"Failed to send request to server: {SERVER_URL}."
    server_file = response["file"]
    if server_file is None:
        file_name = os.path.basename(file_uri)
        response = upload_file(f"{SERVER_URL}/create_file_by_upload_directio", file_uri, dict(
            file_name = file_name,
            folder = remote_virtual_folder,
            description = description
        ))
        assert response is not None, f"Failed to upload file to server: {SERVER_URL}."
        server_file = response["file"]
    return server_file

def create_view(
    view_name, onnx_model, profile0, profile1, remote_virtual_folder="default", description="no description"
):
    onnx_model_file         = upload_file_if_not_exist(onnx_model, remote_virtual_folder=f"{remote_virtual_folder}/onnx_model", description=f"onnx_model for view: {view_name}")
    profile0_layerinfo_file = upload_file_if_not_exist(profile0["layerinfo_file"], remote_virtual_folder=f"{remote_virtual_folder}/profile0", description=f"layerinfo_file for view: {view_name}")
    profile0_profile_file   = upload_file_if_not_exist(profile0["profile_file"], remote_virtual_folder=f"{remote_virtual_folder}/profile0", description=f"profile_file for view: {view_name}")
    profile0_build_log_file = upload_file_if_not_exist(profile0["build_log_file"], remote_virtual_folder=f"{remote_virtual_folder}/profile0", description=f"build_log_file for view: {view_name}")
    profile1_layerinfo_file = upload_file_if_not_exist(profile1["layerinfo_file"], remote_virtual_folder=f"{remote_virtual_folder}/profile1", description=f"layerinfo_file for view: {view_name}")
    profile1_profile_file   = upload_file_if_not_exist(profile1["profile_file"], remote_virtual_folder=f"{remote_virtual_folder}/profile1", description=f"profile_file for view: {view_name}")
    profile1_build_log_file = upload_file_if_not_exist(profile1["build_log_file"], remote_virtual_folder=f"{remote_virtual_folder}/profile1", description=f"build_log_file for view: {view_name}")

    view = request_info(f"{SERVER_URL}/create_view", data=dict(
        view_type="onnx",
        model=onnx_model_file["file_id"],
        layerinfo=profile0_layerinfo_file["file_id"],
        profile=profile0_profile_file["file_id"],
        buildlog=profile0_build_log_file["file_id"],
        trtperf=None,
        layerinfo_compared=profile1_layerinfo_file["file_id"],
        profile_compared=profile1_profile_file["file_id"],
        buildlog_compared=profile1_build_log_file["file_id"],
        trtperf_compared=None,
        with_kernel_match_report=profile0_layerinfo_file["file_id"] is not None and profile1_layerinfo_file["file_id"] is not None,
        folder=remote_virtual_folder,
        description=description,
        name=view_name,
        ignore_cache=False
    ))
    assert view is not None, f"Failed to create view: {view_name}"
    view_id = view["view_id"]
    return f"{SERVER_URL}/view/{view_id}"

def create_trex(
    view_name, layers, profile, remote_virtual_folder="default", description="no description"
):
    layers_file  = upload_file_if_not_exist(layers, remote_virtual_folder=f"{remote_virtual_folder}", description=f"layerinfo_file for view: {view_name}")
    profile_file = upload_file_if_not_exist(profile, remote_virtual_folder=f"{remote_virtual_folder}", description=f"profile for view: {view_name}")

    view = request_info(f"{SERVER_URL}/create_view", data=dict(
        view_type="trex",
        layerinfo=layers_file["file_id"],
        profile=profile_file["file_id"],
        folder=remote_virtual_folder,
        description=description,
        name=view_name
    ))
    assert view is not None, f"Failed to create view: {view_name}"
    view_id = view["view_id"]
    return f"{SERVER_URL}/view/{view_id}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Create View")
    subparser = parser.add_subparsers(dest="type")
    create_onnx_view = subparser.add_parser("onnx")

    # python create_view.py onnx <onnx_file> --layers <layers_file> --profile <profile_file> --log <log_file> --name <view_name> --folder <folder> --desc <description>
    create_onnx_view.add_argument("onnx", type=str)
    create_onnx_view.add_argument("--layers", type=str)
    create_onnx_view.add_argument("--profile", type=str)
    create_onnx_view.add_argument("--log", type=str)
    create_onnx_view.add_argument("--layers2", type=str)
    create_onnx_view.add_argument("--profile2", type=str)
    create_onnx_view.add_argument("--log2", type=str)
    create_onnx_view.add_argument("--name", type=str)
    create_onnx_view.add_argument("--folder", type=str, default="default")
    create_onnx_view.add_argument("--desc", type=str, default="no description")

    # python create_view.py trex <layers_file> --profile <profile_file> --name <view_name> --folder <folder> --desc <description>
    create_trex_view = subparser.add_parser("trex")
    create_trex_view.add_argument("layers", type=str)
    create_trex_view.add_argument("--profile", type=str)
    create_trex_view.add_argument("--name", type=str)
    create_trex_view.add_argument("--folder", type=str, default="default")
    create_trex_view.add_argument("--desc", type=str, default="no description")
    args = parser.parse_args()

    if args.type not in ["trex", "onnx"]:
        parser.print_help()
        exit(1)

    if args.type == "onnx":
        if args.name is None:
            args.name = os.path.basename(args.onnx)
    elif args.type == "trex":
        if args.name is None:
            args.name = os.path.basename(args.layers)

    if args.type == "trex":
        url = create_trex(
            args.name,
            args.layers, args.profile, args.folder, args.desc
        )
    elif args.type == "onnx":
        url = create_view(
            args.name,
            args.onnx,
            {
                "layerinfo_file": args.layers,
                "profile_file": args.profile,
                "build_log_file": args.log,
            },
            {
                "layerinfo_file": args.layers2,
                "profile_file": args.profile2,
                "build_log_file": args.log2,
            },
            args.folder, args.desc
        )
    print(url)