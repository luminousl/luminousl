import os
from datetime import datetime
from subprocess import STDOUT, check_output
from tempfile import NamedTemporaryFile
from dot_dag import generate_graph_svg

def now():
    return datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")

class RemoteExecutor:
    def __init__(self, host, user, password, workdir):
        self.host = host
        self.user = user
        self.password = password
        self.workdir = workdir
        self.pipeline = []

    def ping(self):
        print(f"Pinging {self.host}")
        return self.cmd("echo \"ping message: hello\"", 5).strip().endswith("ping message: hello")

    def cmd(self, command, timeout):
        command = f"sshpass -p {self.password} ssh -o StrictHostKeyChecking=no {self.user}@{self.host} '{command}'"
        print(f"Executing {command}")
        output = check_output(command, stderr=STDOUT, timeout=timeout, shell=True)
        return output.decode("utf-8").strip()
    
    def scp_upload(self, remote, local, timeout):
        remote = os.path.join(self.workdir, remote)
        print(f"Uploading {local} to {remote}")
        command = f"sshpass -p {self.password} scp -o StrictHostKeyChecking=no {local} {self.user}@{self.host}:{remote}"
        check_output(command, stderr=STDOUT, timeout=timeout, shell=True)

    def scp_download(self, remote, local, timeout):
        remote = os.path.join(self.workdir, remote)
        print(f"Downloading {remote} to {local}")
        command = f"sshpass -p {self.password} scp -o StrictHostKeyChecking=no {self.user}@{self.host}:{remote} {local}"
        check_output(command, stderr=STDOUT, timeout=timeout, shell=True)

class ProfilingPipeline:
    def __init__(self, pipeline_script_code, input_files, local_work_folder, tmp_folder="./"):
        self.pipeline_script_code = pipeline_script_code
        self.input_files = input_files
        self.local_work_folder = local_work_folder
        self.local_outputs_folder = os.path.join(self.local_work_folder, "outputs")
        self.tmp_folder = tmp_folder
        os.makedirs(self.local_outputs_folder, exist_ok=True)

        for i in range(len(self.input_files)):
            assert os.path.exists(self.input_files[i]), f"Input file {self.input_files[i]} does not exist"
            self.input_files[i] = os.path.abspath(self.input_files[i])

        self.parse_script_code()

    def parse_script_code(self):
        specific_variables = {}
        for line in self.pipeline_script_code.split("\n"):
            if line.startswith("# "):
                p = line.find(" ", 2)
                if p != -1:
                    name = line[2:p]
                    value = line[p+1:]
                    specific_variables[name] = value

        for name, value in specific_variables.items():
            if name == "WORKDIR":
                value = value.replace("{TIME_NOW}", now())
                specific_variables[name] = value

        self.host_ip = specific_variables.get("HOSTIP")
        self.host_user = specific_variables.get("HOSTUSER")
        self.host_password = specific_variables.get("HOSTPASSWD")
        self.workdir = specific_variables.get("WORKDIR")
        self.layer_info_file = specific_variables.get("LAYER_INFO_JSON", "outputs/layer_info.json")
        self.profile_file = specific_variables.get("PROFILE_JSON", "outputs/profile.json")
        self.build_log_file = specific_variables.get("BUILD_LOG", "outputs/build.log")
        self.remove_workdir = specific_variables.get("CLEANUP", "True").lower() == "true"
        trex_arguments = specific_variables.get("TREX_VIEW", "outputs/layer_info.json outputs/profile.json trex_view.svg")

        required_fields = ["host_ip", "host_user", "host_password", "workdir"]
        for field in required_fields:
            if getattr(self, field) is None:
                raise ValueError(f"Missing required field: {field}")

        self.trex_view = None
        if trex_arguments != "False":
            trex_view_list = trex_arguments.split(" ", maxsplit=2)
            assert len(trex_view_list) >= 1 and len(trex_view_list) <= 3, f"Invalid number of arguments for TREX_VIEW"
            layers = trex_view_list[0] if len(trex_view_list) > 0 else None
            profile = trex_view_list[1] if len(trex_view_list) > 1 else None
            trex_name = trex_view_list[2] if len(trex_view_list) > 2 else "Trex View"
            assert layers is not None, f"Invalid trex view argument"
            self.trex_view = dict(layers=layers, profile=profile, name=trex_name)

    def run(self, status_update=None):
        self.executor = RemoteExecutor(self.host_ip, self.host_user, self.host_password, self.workdir)

        status_update(f"Ping remote device: {self.host_ip}")
        assert self.executor.ping(), f"Failed to ping {self.host_ip}"

        if status_update is None:
            status_update = lambda x: x

        with NamedTemporaryFile(dir=self.tmp_folder) as script_file:
            script_file.write(self.pipeline_script_code.encode("utf-8"))
            script_file.flush()

            status_update(f"Creating work folder: {self.workdir}")
            self.executor.cmd(f"mkdir -p {self.workdir}", 5)
            self.executor.cmd(f"mkdir -p {self.workdir}/inputs", 5)
            self.executor.cmd(f"mkdir -p {self.workdir}/outputs", 5)
            self.executor.scp_upload(f"__profiling_script.sh", script_file.name, 5)

            for file in self.input_files:
                name = os.path.basename(file)
                status_update(f"Copying input file {name} to inputs")
                self.executor.scp_upload(f"inputs/{name}", file, 60 * 10)

            try:
                status_update(f"Executing pipeline...")
                self.executor.cmd(f"cd {self.workdir} && " + " PS4=\"$> \"  bash -x __profiling_script.sh > outputs/console.log 2>&1", 60 * 20)
            except Exception as e:
                pass

            status_update(f"Fetching output files.")
            output_files = self.executor.cmd(f"ls {self.workdir}/outputs", 5).split("\n")
            output_files = list(filter(lambda x: x.strip() != "", output_files))
            output_files = [dict(name=file.strip(), type="plain-text") for file in output_files]
            output_files_full_path = [os.path.join(self.workdir, "outputs", file["name"]) for file in output_files]

            for file in output_files_full_path:
                name = os.path.basename(file)
                status_update(f"Downloading output file: {name}")
                self.executor.scp_download(file, os.path.join(self.local_outputs_folder, name), 60 * 10)
            
            if self.remove_workdir:
                status_update(f"Cleaning the work folder...")
                self.executor.cmd(f"rm -rf {self.workdir}", 60)

        if self.trex_view is not None:
            layers_file = os.path.join(self.local_work_folder, self.trex_view["layers"])
            profile_file = os.path.join(self.local_work_folder, self.trex_view["profile"]) if self.trex_view["profile"] is not None else None

            if os.path.exists(layers_file):
                trex_name = self.trex_view["name"]
                svg_output_file = os.path.join(self.local_outputs_folder, trex_name)
                status_update(f"Creating trex view [{trex_name}]...")
                if generate_graph_svg(layers_file, profile_file, svg_output_file):
                    output_files.insert(0, dict(name=trex_name, type="svg"))
        
        return output_files
