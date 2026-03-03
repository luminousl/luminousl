
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from trex import *

import argparse
import plotly.offline as pyo

import shutil
from parse_trtexec_log import parse_build_log, parse_profiling_log

import glob

from profile2graph import profile2graph
from plan2info_dict import * 

# plan = EnginePlan(graph_file,
#                       profile_file,
#                       profile_meta_file)
