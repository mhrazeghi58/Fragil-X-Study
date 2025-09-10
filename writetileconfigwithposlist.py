import json
import numpy as np
import glob
from pyometiff import OMETIFFWriter


def get_pos_array(poslist):
    return poslist["map"]["StagePositions"]["array"]


def get_xy_stage(poslist):
    check_uniform_xy_stage(poslist)
    return poslist["map"]["StagePositions"]["array"][0]["DefaultXYStage"]["scalar"]


def get_z_stage(poslist, allow_blank=False):
    check_uniform_z_stage(poslist, allow_blank=allow_blank)
    return poslist["map"]["StagePositions"]["array"][0]["DefaultZStage"]["scalar"]


def check_uniform_xy_stage(poslist):
    check_uniform_value(get_pos_array(poslist), lambda p: p["DefaultXYStage"]["scalar"])


def check_uniform_z_stage(poslist, allow_blank=False):
    check_uniform_value(
        get_pos_array(poslist),
        lambda p: p["DefaultZStage"]["scalar"],
        blank_value="" if allow_blank else None,
    )


def check_uniform_value(array, value_getter, blank_value=None):
    values = set(value_getter(v) for v in array)
    if len(values) == 1:
        return
    if blank_value is not None and len(values) == 2 and blank_value in values:
        return
    raise ValueError(f"Non-uniform value: {list(values)}")


def find_xy_coord_of_pos(p, xy_stage):
    devposs = p["DevicePositions"]["array"]
    for devpos in devposs:
        if devpos["Device"]["scalar"] == xy_stage:
            xy = devpos["Position_um"]["array"]
            if len(xy) != 2:
                raise ValueError("XY position not 2D")
            return xy
    raise ValueError("Missing xy")


def find_z_coord_of_pos(p, z_stage, create_if_missing=False):
    devposs = p["DevicePositions"]["array"]
    for devpos in devposs:
        if devpos["Device"]["scalar"] == z_stage:
            z = devpos["Position_um"]["array"]
            if len(z) != 1:
                raise ValueError("Z position not scalar")
            return z
    if create_if_missing:
        devposs.append(
            {
                "Device": {"type": "STRING", "scalar": z_stage},
                "Position_um": {"type": "DOUBLE", "array": [None]},
            }
        )
        return devposs[-1]["Position_um"]["array"]
    raise ValueError("Missing z")


def get_xy_positions(poslist, xy_stage):
    ps = get_pos_array(poslist)
    coords = []
    for p in ps:
        xy = find_xy_coord_of_pos(p, xy_stage)
        coords.append((xy[0], xy[1]))
    return np.array(coords)


def get_xyz_positions(poslist, xy_stage, z_stage):
    ps = get_pos_array(poslist)
    coords = []
    for p in ps:
        try:
            xy, z = find_xy_coord_of_pos(p, xy_stage), find_z_coord_of_pos(p, z_stage)
        except ValueError as e:
            continue
        coords.append((xy[0], xy[1], z[0]))
    return np.array(coords)



def create_tile_config(outputpath, prefix,x,y):
    filename = f"{prefix}"
    
    #print(f"filename: {filename.split("/")[1]}")
   
    tile_config_row = (f"{filename}; ; ({x},{y})")
    #print(tile_config_row)
    write_tile_config(outputpath, tile_config_row,filename.split("/")[-1])
    return 

def write_tile_config(outputpath, tile_config_row,position):
    with open(f"{outputpath}/tile_config.txt","a") as text_file:
        text_file.write(tile_config_row+ "\n")
        

def main():
   
    input_filepath = 'E:\Code\MM_bigfovacqusition\democoords_zcalculated.pos'
    imagelocation = 'E:\\Code\\MM_bigfovacqusition\\tiffimages\\'
    filelist = []
    for files in glob.glob(imagelocation +'*.tif'):
        filelist.append(files)
    print(f'this is filelist{filelist}')
    outputpath = 'E:\\Code\\MM_bigfovacqusition\\'
    
    with open(input_filepath) as f:
        poslist = json.load(f)

    xy_stage = get_xy_stage(poslist)
    z_stage = get_z_stage(poslist, allow_blank=True)
    xyzs = get_xyz_positions(poslist, xy_stage, z_stage)
    print(xyzs)
    x = xyzs[:, 0]
    y = xyzs[:, 1]
    z = xyzs[:, 2]
    print(f'this is x{x}')
    idx = 0
    for i in filelist:
        name = i
        create_tile_config(outputpath, name,x[idx],y[idx])
        idx = idx+1

if __name__ == "__main__":
    main()
