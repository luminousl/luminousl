import os
import sys
import glob
import random
import logging

import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

def get_data_loader(input_data, max_input_size, preprocess_func_name, input_info, batch_size=1):
    # Use input cache if it exists
    if os.path.exists(input_data):
        logger.info(f"Getting input data in {input_data}")
        input_files = get_input_files(input_data, max_input_size)
    else:
        raise ValueError(f"ERROR: input data directory not found in {input_data}")

    # Choose pre-processing function for INT8 input
    import processing
    if preprocess_func_name is not None:
        preprocess_func = getattr(processing, preprocess_func_name)
    else:
        preprocess_func = processing.preprocess_fakedata
    data_loader = FakeInputDataLoader(input_files=input_files,
                                         batch_size=batch_size,
                                         input_info=input_info,
                                         preprocess_func=preprocess_func)
    return data_loader


def get_input_files(input_data, max_input_size=None, allowed_extensions=(".npy")):
    """Returns a list of all filenames ending with `allowed_extensions` found in the `input_data` directory.

    Parameters
    ----------
    input_data: str
        Path to directory containing desired files.
    max_input_size: int
        Max number of files to use for input. If input_data contains more than this number,
        a random sample of size max_input_size will be returned instead. If None, all samples will be used.

    Returns
    -------
    input_files: List[str]
         List of filenames contained in the `input_data` directory ending with `allowed_extensions`.
    """

    logger.info("Collecting input files from: {:}".format(input_data))
#     print("Cali Path:", input_data)
    input_files = [path for path in glob.iglob(os.path.join(input_data, "**"), recursive=True)
                         if os.path.isfile(path) and path.lower().endswith(allowed_extensions)]
    logger.info("Number of Input Files found: {:}".format(len(input_files)))

    if len(input_files) == 0:
        raise Exception("ERROR: Input data path [{:}] contains no files!".format(input_data))

    if max_input_size:
        if len(input_files) > max_input_size:
            logger.warning("Capping number of input files to max_input_size: {:}".format(max_input_size))
            # random.seed(1)  # Set seed for reproducibility
            # input_files = random.sample(input_files, max_input_size)


    return input_files


# https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/infer/Int8/EntropyCalibrator2.html
class FakeInputDataLoader():
    """INT8 Calibrator Class for Imagenet-based Image Classification Models.

    Parameters
    ----------
    input_files: List[str]
        List of image filenames to use for input
    batch_size: int
        Number of images to pass through in one batch during input
    input_shape: Tuple[int]
        Tuple of integers defining the shape of input to the model (Default: (3, 224, 224))
    cache_file: str
        Name of file to read/write input cache from/to.
    preprocess_func: function -> numpy.ndarray
        Pre-processing function to run on input data. This should match the pre-processing
        done at inference time. In general, this function should return a numpy array of
        shape `input_shape`.
    """

    def __init__(self, input_files=[], batch_size=1, input_info={}, preprocess_func=None):
        super().__init__()
        self.input_info = input_info
        self.modified_cache = False
        self.batch_size = batch_size
        self.batch = {k:np.zeros((self.batch_size, *v['shape']), dtype=v['dtype']) for k,v in self.input_info.items()}
        self.files = input_files
        # Pad the list so it is a multiple of batch_size
        if len(self.files) % self.batch_size != 0:
            logger.info("Padding # input files to be a multiple of batch_size {:}".format(self.batch_size))
            self.files += input_files[(len(input_files) % self.batch_size):self.batch_size]

        self.batches = self.load_batches()
        self.no_quant = []

        if preprocess_func is None:
            logger.error("No preprocess_func defined! Please provide one to the constructor.")
            sys.exit(1)
        else:
            self.preprocess_func = preprocess_func

    def load_batches(self):
        # Populates a persistent self.batch buffer with images.
        for index in range(0, len(self.files), self.batch_size):
            for offset in range(self.batch_size):
                # image = Image.open(self.files[index + offset])
                data = np.load(self.files[index + offset], allow_pickle=True).item()
                for k, v in data.items():
                    self.batch[k][offset] = self.preprocess_func(v)
            logger.info("input images pre-processed: {:}/{:}".format(index+self.batch_size, len(self.files)))
            yield self.batch

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            # Assume self.batches is a generator that provides batch data.
            batch = next(self.batches)
            batch_input = []
            for name in names:
                batch_input.append(batch[name])
            return batch_input
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no input data remaining.
            return None