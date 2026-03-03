from trex.graphing import to_dot, layer_type_formatter, layer_colormap
from trex import EnginePlan
import os
import tempfile
import traceback
import datetime

def now():
    return datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

def dprint(msg):
    if not isinstance(msg, str):
        msg = str(msg)

    print(f"{now()}: {msg}", flush=True)

layer_colormap.update({
    "correlation": "#4682B4",
    "kgen": "#34a853",
    "fusion": "#4285f4"
})

def get_engine_dag(graph_file, profile_file=None):
    plan = EnginePlan(graph_file, profile_file)
    formatter = layer_type_formatter
    display_regions = True
    expand_layer_details = False
    graph = to_dot(plan, formatter,
                display_regions=display_regions,
                expand_layer_details=expand_layer_details)
    return "".join(graph)

def generate_graph_svg(layers, profiles, output):
    try:
        dprint(f"Make dot svg {layers} to {output}")
        dag = get_engine_dag(layers, profiles)
        dprint(f"Make dot svg {layers} to {output} done.")
        with tempfile.NamedTemporaryFile("w", delete=False) as fp:
            fp.write(dag)
            fp.flush()
            dprint(f"dot -Tsvg -o \"{output}\" {fp.name}")
            code = os.system(f"/usr/bin/dot -Tsvg {fp.name} -o\"{output}\"")
            assert code == 0, f"Failed to call: dot -Tsvg -o {output} {fp.name}"
        dprint(f"===== Make dot svg done.")
    except Exception as e:
        dprint(traceback.format_exc())
        return False
    return True
