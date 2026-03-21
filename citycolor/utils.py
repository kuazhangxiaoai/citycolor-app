import os

import cv2
import yaml

from citycolor.config import Config
import numpy as np
from sklearn.cluster import KMeans
from citycolor.convert import convert_rgb2hsv, convert_hsv2rgb, rgb_to_hsv, hsv_to_rgb

faces = ['px', 'nx', 'py', 'ny', 'pz', 'nz']

def get_main_name(filepath):
    return filepath.split(os.sep)[-1].split('.')[0]

def clustering(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    hsv = kmeans.cluster_centers_.astype(np.float32)
    counts = np.bincount(kmeans.labels_) / data.shape[0]
    return counts, hsv

def get_dynamic_k(color_hist, saturate_hist, value_hist, thresh=0.02):
    nc = (color_hist > thresh).sum()
    ns = (saturate_hist > thresh).sum()
    nv = (value_hist > thresh).sum()

    return np.sqrt(nc * ns) + np.sqrt(nc * nv) + np.sqrt(nv * ns)

def hue_analytic(image: np.ndarray, mask: np.ndarray, config:Config):
    h, w, _ = image.shape
    color_hist = [0] * config.color_degree_num
    img = image[mask] if mask is not None else image
    #values_hist = [0] * config.value_degree_num
    for pixel in img:
        b, g, r = pixel
        h, s, v = rgb_to_hsv(r, g, b)
        ha = int((h - 1e-9) / config.color_resolution) if h > 0 else 0
        color_hist[ha] += 1
        # va = int((v - 1e-9) / config.value_resolution) if v > 0 else 0
        # values_hist[va] += 1

    return np.array(color_hist) / (h*w) #config.values[np.array(values_hist).argmax()]

def saturation_analytic(image: np.ndarray, mask: np.ndarray, hue_index, config:Config):
    h, w, _ = image.shape
    saturation_hist = [0] * config.saturation_degree_num
    img = image[mask] if mask is not None else image
    for pixel in img:
        b, g, r = pixel
        h, s, v = rgb_to_hsv(r, g, b)
        ha = int((h - 1e-9) / config.color_resolution) if h > 0 else 0
        sa = int((s - 1e-9) / config.saturation_resolution) if s > 0 else 0
        if ha == hue_index:
            saturation_hist[sa] += 1

    return np.array(saturation_hist) / (h * w)

def normalize(v):
    return v / np.linalg.norm(v, axis=2, keepdims=True)

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def find_category_index(name, cfg):
    names, colors = cfg["names"], cfg["colors"]
    for k in names.keys():
        if names[k] == name:
            return k, colors[k]
    return -1, None

def find_mask(colormap, color):
    r, g, b = color
    mb, mg, mr = colormap[:,:,0] == b, colormap[:,:,1] == g, colormap[:,:,2] == r
    return mb * mg * mr

def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles

def pano2cube(image: np.ndarray, imgsz=512):
    h, w,_ = image.shape
    cube = {}

    for f in faces:
        u = np.linspace(-1,1, imgsz)
        v = np.linspace(-1,1, imgsz)
        uu, vv = np.meshgrid(u, v)
        if f == 'px':
            x = np.ones_like(uu)
            y = -vv
            z = -uu
        elif f == 'nx':
            x = -np.ones_like(uu)
            y = -vv
            z = uu
        elif f == 'py':
            x = uu
            y = np.ones_like(uu)
            z = vv
        elif f == 'ny':
            x = uu
            y = -np.ones_like(uu)
            z = -vv
        elif f == 'pz':
            x = uu
            y = -vv
            z = np.ones_like(uu)
        elif f == 'nz':
            x = -uu
            y = -vv
            z = -np.ones_like(uu)
        vec = normalize(np.stack([x, y, z], axis=-1))
        dx, dy,dz = vec[...,0], vec[...,1],vec[...,2]
        thelta = np.arctan2(dz, dx)
        phi = np.arccos(dy)

        uf = (thelta + np.pi) / (2 * np.pi) * (w - 1)
        vf = phi / np.pi * (h - 1)
        mapx, mapy = uf.astype(np.float32), vf.astype(np.float32)
        fimg = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        cube[f] = fimg
    return cube



