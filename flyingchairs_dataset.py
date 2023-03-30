# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

from typing import List, Optional
import random
from glob import glob
import os.path as osp
from os.path import splitext

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import cv2


def readFlow(fn):
    """Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print("Magic number incorrect. Invalid .flo file")
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def readPFM(file):
    file = open(file, "rb")

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b"PF":
        color = True
    elif header == b"Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(rb"^(\d+)\s(\d+)\s$", file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == ".png" or ext == ".jpeg" or ext == ".ppm" or ext == ".jpg":
        return Image.open(file_name)
    elif ext == ".bin" or ext == ".raw":
        return np.load(file_name)
    elif ext == ".flo":
        return readFlow(file_name).astype(np.float32)
    elif ext == ".pfm":
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []


class MyFlyingChairs(data.Dataset):
    def __init__(
        self,
        split="train",
        root="datasets/FlyingChairs_release/data",
        scale_factor: float = 1.0,
        preds_path: Optional[str] = None,
        normalize: bool = False,
        to_gray: bool = False,
    ):
        split_list = np.loadtxt(osp.join(root, "FlyingChairs_train_val.txt"), dtype=np.int32)

        images = sorted(glob(osp.join(root, "data/*.ppm")))
        flows = sorted(glob(osp.join(root, "data/*.flo")))
        assert len(images) // 2 == len(flows)

        if preds_path is not None:
            preds = sorted(glob(osp.join(preds_path, "*.flo")))
            self.preds_list = []
            assert len(preds) == len(flows)
        else:
            preds = None
            self.preds_list = None

        self.flow_list = []
        self.image_list = []
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == "train" and xid == 1) or (split == "validation" and xid == 2):
                self.flow_list.append(flows[i])
                self.image_list += [[images[2 * i], images[2 * i + 1]]]
                if preds and self.preds_list is not None:
                    self.preds_list.append(preds[i])

        self.init_seed = False
        self.scale = scale_factor
        self.normalize = normalize
        self.to_gray = to_gray

    def __getitem__(self, index):
        if not self.init_seed:
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            self.init_seed = True
        index = index % len(self.image_list)
        valid = None

        flow = read_gen(self.flow_list[index])
        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])
        pred = read_gen(self.preds_list[index]) if self.preds_list else None

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)[..., :3]
        img2 = np.array(img2).astype(np.uint8)[..., :3]
        pred = np.array(pred).astype(np.float32) if pred is not None else None

        if self.scale != 1.0:
            img1 = cv2.resize(img1, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
            flow = flow * self.scale
            pred = (
                cv2.resize(
                    pred,
                    None,
                    fx=self.scale,
                    fy=self.scale,
                    interpolation=cv2.INTER_LINEAR,
                )
                if pred is not None
                else None
            )
            pred = pred * self.scale if pred is not None else None

        img1_gray = img2_gray = None
        if self.to_gray:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            img1_gray = torch.from_numpy(img1_gray).unsqueeze(-1).permute(2, 0, 1).float()
            img2_gray = torch.from_numpy(img2_gray).unsqueeze(-1).permute(2, 0, 1).float()

        img1 = torch.from_numpy(img1).float()  # .permute(2, 0, 1)
        img2 = torch.from_numpy(img2).float()  # .permute(2, 0, 1)
        flow = torch.from_numpy(flow).float()  # .permute(2, 0, 1)
        pred = torch.from_numpy(pred).float() if pred is not None else None

        if self.normalize:
            img1 = img1 / 255  # Normalize to [0,1]
            img2 = img2 / 255  # Normalize to [0,1]
            if self.to_gray:
                img1_gray = (img1_gray / 255) * 2 - 1
                img2_gray = (img2_gray / 255) * 2 - 1
            # Assume the range of flow values in FlyingChairs is [-300,300]
            flow = flow / (300 * self.scale)
            pred = pred / (300 * self.scale) if pred is not None else None

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        meta = {
            "img1_path": self.image_list[index][0],
            "img2_path": self.image_list[index][1],
            "flow_path": self.flow_list[index],
            "pred_path": self.preds_list[index] if self.preds_list else "",
        }
        flow_3ch = torch.cat((flow, flow[..., -1].unsqueeze(-1)), -1)  # Extra channel for the autoencoder
        out_dict = {"jpgs": torch.cat((img1, img2), -1), "flow": flow_3ch, "txt": ""}

        return out_dict

    def __len__(self):
        return len(self.image_list)
