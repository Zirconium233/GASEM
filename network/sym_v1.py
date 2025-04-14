import torch
from typing import List
from datasets.GAPartNet.misc.info import OBJECT_NAME2ID
from datasets.datasets_pair import PointCloudPair
import numpy as np

ID2OBJECT_NAME = {v: k for k, v in OBJECT_NAME2ID.items()}

def get_sym_from_input(pc_pairs: List[PointCloudPair]):
    sym_list_1 = []
    sym_list_2 = []
    for pc_pair in pc_pairs:
        sym_1 = get_sym_info(pc_pair.pc1.obj_cat)
        sym_2 = get_sym_info(pc_pair.pc2.obj_cat)
        sym_list_1.append(sym_1)
        sym_list_2.append(sym_2)
    sym1 = np.stack(sym_list_1)
    sym2 = np.stack(sym_list_1)
    return sym1, sym2

def get_sym_info(object_name, mug_handle=1):
    # Define symmetry information for each object type
    # sym_info: c0: face classification, c1, c2, c3: symmetry in xy, xz, yz planes respectively
    object_name = ID2OBJECT_NAME[object_name]
    sym_info = {
        "Box": np.array([1, 1, 1, 1], dtype=np.int32),  # Cube symmetry
        "Remote": np.array([0, 0, 0, 0], dtype=np.int32),
        "Microwave": np.array([0, 0, 0, 0], dtype=np.int32),
        "Camera": np.array([0, 0, 0, 0], dtype=np.int32),
        "Dishwasher": np.array([0, 0, 0, 0], dtype=np.int32),
        "WashingMachine": np.array([0, 0, 0, 0], dtype=np.int32),
        "CoffeeMachine": np.array([0, 0, 0, 0], dtype=np.int32),
        "Toaster": np.array([0, 0, 0, 0], dtype=np.int32),
        "StorageFurniture": np.array([0, 0, 0, 0], dtype=np.int32),
        "AKBBucket": np.array([1, 1, 1, 1], dtype=np.int32),
        "AKBBox": np.array([1, 1, 1, 1], dtype=np.int32),
        "AKBDrawer": np.array([0, 0, 0, 0], dtype=np.int32),
        "AKBTrashCan": np.array([1, 1, 0, 1], dtype=np.int32),
        "Bucket": np.array([1, 1, 0, 1], dtype=np.int32),
        "Keyboard": np.array([0, 0, 0, 0], dtype=np.int32),
        "Printer": np.array([0, 0, 0, 0], dtype=np.int32),
        "Toilet": np.array([0, 0, 0, 0], dtype=np.int32),
        "KitchenPot": np.array([1, 1, 0, 1], dtype=np.int32),
        "Safe": np.array([0, 0, 0, 0], dtype=np.int32),
        "Oven": np.array([0, 0, 0, 0], dtype=np.int32),
        "Phone": np.array([0, 0, 0, 0], dtype=np.int32),
        "Refrigerator": np.array([0, 0, 0, 0], dtype=np.int32),
        "Table": np.array([0, 0, 0, 0], dtype=np.int32),
        "TrashCan": np.array([1, 1, 0, 1], dtype=np.int32),
        "Door": np.array([0, 0, 0, 0], dtype=np.int32),
        "Laptop": np.array([0, 1, 0, 0], dtype=np.int32),
        "Suitcase": np.array([0, 0, 0, 0], dtype=np.int32)
    }
    
    # Handle mugs separately as it has a parameter for the handle
    if object_name == 'mug' and mug_handle == 1:
        return np.array([0, 1, 0, 0], dtype=np.int32)
    elif object_name == 'mug' and mug_handle == 0:
        return np.array([1, 0, 0, 0], dtype=np.int32)
    
    # Return the symmetry information for the object, or default to no symmetry if not found
    return sym_info.get(object_name, np.array([0, 0, 0, 0], dtype=np.int32))

def get_sym_info_part(part_name):
    #  sym_info  c0 : face classification  c1, c2, c3: Three view symmetry, correspond to xy, xz, yz respectively
    #  c0: 0 no symmetry 1 axis symmetry 2 two reflection planes 3 unimplemented type

    if part_name == 'others':
        sym = np.array([0, 0, 0, 0], dtype=np.int32)
    elif part_name == 'line_fixed_handle':
        sym = np.array([1, 1, 0, 1], dtype=np.int32)
    elif part_name == 'round_fixed_handle':
        sym = np.array([1, 1, 0, 1], dtype=np.int32)
    elif part_name == 'slider_button':
        sym = np.array([0, 0, 0, 0], dtype=np.int32)
    elif part_name == 'hinge_door':
        sym = np.array([2, 0, 1, 1], dtype=np.int32)
    elif part_name == 'slider_drawer':
        sym = np.array([1, 1, 0, 1], dtype=np.int32)
    elif part_name == 'slider_lid':
        sym = np.array([0, 0, 0, 0], dtype=np.int32)
    elif part_name == 'hinge_lid':
        sym = np.array([2, 0, 1, 1], dtype=np.int32)
    elif part_name == 'hinge_knob':
        sym = np.array([1, 1, 0, 1], dtype=np.int32)
    elif part_name == 'revolute_handle':
        sym = np.array([1, 1, 0, 1], dtype=np.int32)
    else:
        sym = np.array([0, 0, 0, 0], dtype=np.int32)
    return sym

