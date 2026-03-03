import os
import struct 
import json
import sys
from collections import OrderedDict
import logging

# calib_path = "/data/project/zdrive_develop_prod/bevformer/make_trt/seq/CalibrationTable_slim_head_lstm_seq_sim"
# calib_path = "/data/project/zdrive_develop_prod/bevformer/make_trt/src/CalibrationTable_checkpoint_v7_alldata_cls4_v1_ep8_sim_dla_delete_v4"




namemaps = []

def read_calibtable_txt2json(calib_file):
    logging.info("calibration input txt: %s" % (calib_file,))
    root = OrderedDict()
    index_name = 0
    index_val = 1
    with open(calib_file) as calibtxt:
        for line in calibtxt:

            line = line.strip()
            logging.debug("")
            logging.debug(">> %s" % line)

            # colon delimited
            fields = line.split(': ')
            # print(fields)
            if ": " not in line:
                continue
            # if len(fields) != 2:
            #     logging.debug("...skipping")
            #     continue

            layername = fields[index_name].strip()
            float_hex = fields[-1].strip()
            float_val = struct.unpack('>f', bytes.fromhex(float_hex))[0]
            if float_val > 1:
                print("%s: %s", layername, float_val)

            vald = OrderedDict()
            vald['scale'] = float_val
            # vald['min'] = 0
            # vald['max'] = 0
            # vald['offset'] = 0
            root[layername] = vald
    return root

def dump_json(json_root, json_file):
    logging.info("calibration output json: %s" % (json_file,))
    json_str = json.dumps(json_root, indent=4)
    with open(json_file, 'w') as fp:
        fp.write(json_str)
        logging.debug(json_str)

if __name__ == "__main__":
    logging.basicConfig(
                format='%(levelname)s: %(message)s',
                level=logging.INFO)
                #level=logging.DEBUG)
    calib_file = "/media/models/vision3/orin/CalibrationTable_checkpoint_7b80m_iter11k_plugin_sim"
    json_file = "/media/models/vision3/orin/CalibrationTable_checkpoint_7b80m_iter11k_plugin_sim.txt"

    json_data = read_calibtable_txt2json(calib_file)
    dump_json(json_data, json_file)
