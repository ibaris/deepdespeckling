import glob
import json
import os
import shutil
import typing as typ

import colorama as cama
import numpy as np
from haightpy.io import write_raster
from haightpy.io.read_config import read_config
from osgeo import gdal
from torchgeo import load_radian_image
from torchgeo.io import TIF as TIFWriter
from torchpy import modules

from deepdespeckling.merlin.test.load_cosar import cos2mat
from deepdespeckling.merlin.test.model import *
from deepdespeckling.merlin.test.model_test import *
from deepdespeckling.merlin.test.utils import *

CONFIG = read_config()


def deepdespeckle(radian_input: str,
                  original_file: typ.Optional[str] = None,
                  output: typ.Optional[str] = None,
                  suffix: typ.Optional[str] = None,
                  prefix: typ.Optional[str] = None,
                  weights_path: typ.Optional[str] = None,
                  stride: int=64 * 3,
                  device="cuda:1"):
    """Use the Merlin Filter and filter the inputted radar image.

    Parameters
    ----------
    radian_input : str
        Radian image path.
    original_file : typ.Optional[str], optional
        Path to the original file, by default None. If none, the path in the radian metadata will be used.
    mesh_input : str
        Mesh object path.
    output_dir : typ.Optional[str]
        Path to the output directory.
    suffix : typ.Optional[str]
        Suffix for the output file.
    prefix : typ.Optional[str]
        A prefix for the output file.
    weights_path : typ.Optional[str], optional
        Path to the merlin weights, by default None. If None, the path specified in the config will be used.
    stride : int, optional
        Stride value, by default 64 * 3.
    device : str, optional
        Define the GPU device, by default "cuda:1"

    Raises
    ------
    AssertionError
        _description_
    """
    # ----------------------------------------------------------------------------------------------
    # Utility Variables
    # ----------------------------------------------------------------------------------------------
    radian_image = load_radian_image(radian_input)
    image_path = original_file if original_file is not None else radian_image["original_image_file"]
    basename = os.path.basename(radian_input).split(".")[0]

    # Make a temporary path for the Merlin filter. There are too much outputted things.
    tmp_output = os.path.join(output, ".tmp")
    if not os.path.exists(tmp_output):
        os.makedirs(tmp_output)

    # Merlin Variables =================================================================
    if weights_path is None:
        weights_path = CONFIG["merlin"]["weights"]

    if radian_image["SAR_mode"] == "ST":
        weights_path = os.path.join(weights_path, "spotlight.pth")

    elif radian_image["SAR_mode"] == "SM":
        weights_path = os.path.join(weights_path, "stripmap.pth")
    else:
        raise AssertionError("Only Sportlight, Stering Sportlight and Stripmap data can be filtered.")

    # ----------------------------------------------------------------------------------------------
    # Run Merlin
    # ----------------------------------------------------------------------------------------------
    denoiser = Denoiser()

    image_data = cos2mat(image_path)

    denoised_image = denoiser.test(images = [image_data],
                                   stride=64 * 3,
                                   weights_path=weights_path,
                                   patch_size=256, 
                                   device=device)[0]

    # Move Results =====================================================================
    suffix = "_" + suffix if suffix is not None else ""
    prefix = "_" + prefix if prefix is not None else ""

    new_path = os.path.join(output, f"Merlin{prefix}_{basename}{suffix}.tif")

    # metadata = {"radian": json.dumps(img)}
    writer = TIFWriter(filename=new_path, image=radian_image, nodata=-99999)
    writer.create()
    writer.write(denoised_image)
    writer.finish_raster()



RADIAN_INPUT = r"F:\Daten\Haight\data\external\SAR\SAR\TDX_ST_20190530-052615_VV.tif"
ORIGINAL_FILE = r"F:\Daten\Haight\data\external\SAR\SAR\TDX1_SAR__SSC______ST_S_SRA_20190530T052615_20190530T052615\IMAGEDATA\IMAGE_VV_SRA_spot_042.cos"
MESH_INPUT = r"F:\Daten\Haight\2020_Berlin\ry3d\2020_Berlin.ry3d"
OUTPUT = r"F:\Daten\Haight\data\processed\2020_Berlin"
SUFFIX = "TDX_ST"
PREFIX = None
TMP_FOLDER = None

# outputs = make_dataset(radian_input=RADIAN_INPUT,
#                        mesh_input=MESH_INPUT,
#                        output=OUTPUT,
#                        suffix=SUFFIX,
#                        prefix=PREFIX,
#                        tmp_folder=TMP_FOLDER)

deepdespeckle(radian_input=RADIAN_INPUT, original_file=ORIGINAL_FILE, output=OUTPUT, device="cuda:1")
