import json
import os

import cv2
import yaml
from langsmith import expect

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

def get_expect(hist:np.ndarray, dvideby=True):
    xs = np.arange(hist.shape[0])
    ys = hist.astype(np.float32) / hist.sum()
    if dvideby:
        expect = (xs * ys).sum() / hist.shape[0]
    else:
        expect = (xs * ys).sum()
    return expect

def get_image_block(image, mask, blocksize=9):
    h, w, _ = image.shape
    grid_y, grid_x = np.meshgrid(np.arange(start=0, stop=h, step=blocksize), np.arange(start=0, stop=w, step=blocksize), indexing='ij')
    grid = np.stack([grid_y, grid_x], axis=-1).reshape(-1,2)
    blocks = []
    for pt in grid:
        y, x= pt[0], pt[1]
        center = (x + blocksize // 2, y + blocksize // 2)
        if center[0] > w or center[1]> h:
            continue
        if mask[center[1], center[0]]:
            blocks.append(image[y: y + blocksize, x: x + blocksize,:])
    return blocks

def get_hist(image):
    hist = np.array([0] * 256)
    for i in range(256):
        mask = image[image==i]
        hist[i] = mask.sum()
    return hist

def get_strip(image: np.ndarray, strip_num=5):
    h, w = image.shape[:2]
    strips = []
    step = w // strip_num
    strip_infos = []
    for x in range(0, w, step):
        strip = image[:, x: x+step, :]
        strip_info = {
            "p1": [x, 0],
            "p2": [x+step, h]
        }
        strips.append(strip)
        strip_infos.append(strip_info)
    return strips, strip_infos

def rgb_normalize(r, g, b):
    s = float(r) + float(g) + float(b)
    return r / s, g / s, b / s

def get_vertical_color_card(image: np.ndarray, mask: np.ndarray, cardfile: str, stripnum: int, cellsize: int):
    strips,_strip_infos = get_strip(image, stripnum)
    strips_m,_ = get_strip(mask[:,:, None], stripnum)
    stats = []
    for img, msk in zip(strips, strips_m):
        stat = get_color_cards(img, msk[:,:, 0], cardfile, cellsize)
        stats.append(stat)
    return stats, _strip_infos


def get_color_cards(image: np.ndarray, mask: np.ndarray, cardfile: str, blocksize: int, topk=-1):
    with open(cardfile, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    blocks = get_image_block(image, mask, blocksize)
    flag = mask.astype(np.int32)
    if flag.sum() == 0:
        return None
    color = [c["rgb"] for c in cards]
    colors0 = np.array([rgb_normalize(*c["rgb"]) for c in cards])
    names = [c["name"] for c in cards]
    stats = [{"name": name, "color": color, "rgb": cl ,"count": 0} for name, color, cl in zip(names, colors0.tolist(), color)]
    for blk in blocks:
        hist_b, hist_g, hist_r = get_hist(blk[:,:,0]),get_hist(blk[:,:,1]),get_hist(blk[:,:,2])
        #value_max = blk.max(axis=-1).mean()
        #if value_max < 25:
        #    continue
        eb, eg, er = hist_b.argmax(), hist_g.argmax(), hist_r.argmax()
        eh, es, ev = rgb_normalize(er, eg, eb)
        colors1 = np.repeat(np.array([eh, es, ev])[None,:], colors0.shape[0], axis=0)
        diffs = np.abs(colors1 - colors0).sum(axis=-1)
        min_index = diffs.argmin()
        #name = names[min_index]
        stats[min_index]["count"] += (blocksize * blocksize)

    total = 0
    new_stats = []
    for i, stat in enumerate(stats):
        if stats[i]["count"] > 0.05 * len(blocks):
            total = total + stats[i]["count"]

    for i, stat in enumerate(stats):
        if stat["count"] > 0.05 * len(blocks):
            stat["ratio"] = stat["count"] * 1.0 / total
            new_stats.append(stat)
        #stats[i]["ratio"] = stats[i]["count"] * 1.0 / total
    if topk > 0:
        ratio_list = [0] * len(new_stats)
        topk = min(topk, len(new_stats))
        for i in range(len(new_stats)):
            ratio_list[i] = new_stats[i]['ratio']
        idx = np.argpartition(np.array(ratio_list), -topk)[-topk:]
        topk_values = np.array(ratio_list)[idx]
        new_stats_topk = []
        for k in idx:
            new_stat_topk = new_stats[k]
            new_stat_topk['ratio'] = new_stat_topk['ratio'] / topk_values.sum()
            new_stats_topk.append(new_stats[k])
        return new_stats_topk
    else:
        return new_stats













