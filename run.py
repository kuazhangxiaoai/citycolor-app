import os
from pathlib import Path
import numpy as np
import cv2
from citycolor.config import Config
from ultralytics.utils import YAML
from ultralytics import YOLO
from ultralytics.utils.plotting import plot_predict_samples
from citycolor.utils import get_main_name, pano2cube, GetFileFromThisRootDir, load_config, \
    find_category_index, find_mask, hue_analytic, saturation_analytic, get_expect
from citycolor.plotting import plot_hue, plot_saturation, plot_hue_and_sat

project_dir = Path(__file__).resolve().parent
config = Config(project_dir)

def detect(imagepath, save_image_dir, save_mask_dir):
    name = get_main_name(imagepath)
    try:
        model_path = "./weights/yolo11x-semseg.pt"
        data_path = "./ultralytics/cfg/datasets/CityscapesYOLO.yaml"
        model = YOLO(model_path)
        model.overrides["data"] = "./ultralytics/cfg/datasets/CityscapesYOLO.yaml"
        result = model(imagepath)[0]
        save_image_path = os.path.join(save_image_dir, name + '.png')
        save_mask_path = os.path.join(save_mask_dir, name + '.png')
        plot_predict_samples(
            result.orig_img,
            result.masks,
            nc=YAML.load(data_path)["nc"],
            colors=YAML.load(data_path)["colors"],
            fname=save_image_path,
            mname=save_mask_path,
            one_hot=True,
            overlap=False,
        )
        return result.masks
    except Exception as e:
        print(e)

    return None

def main():
    print("CityColor runner start")
    cube_dir = config.cube_dir
    img_paths = GetFileFromThisRootDir(config.img_dir)
    det_cfg = load_config(config.categories_file)
    for img_path in img_paths:  #全景照片路径
        cube = pano2cube(cv2.imread(img_path))
        basename = get_main_name(img_path)
        os.mkdir(os.path.join(cube_dir, basename)) if not os.path.exists(os.path.join(cube_dir, basename)) else None
        save_image_dir = os.path.join(config.save_image_dir, basename)
        save_mask_dir = os.path.join(config.save_mask_dir, basename)
        os.mkdir(save_image_dir) if not os.path.exists(save_image_dir) else None
        os.mkdir(save_mask_dir) if not os.path.exists(save_mask_dir) else None
        for k in cube.keys():
            if k in config.attention_direction:
                cv2.imwrite(os.path.join(cube_dir, basename, k+'.png'), cube[k])
                cube_image_file = os.path.join(cube_dir, basename, k+'.png')
                detect(cube_image_file, save_image_dir, save_mask_dir)
                for category in config.attention_categories:
                    index, color = find_category_index(category, det_cfg)
                    assert index > 0, "Cannot find the categories"
                    colormap = cv2.imread(os.path.join(save_mask_dir, k+'.png'))
                    mask = find_mask(colormap, color)
                    mask_gray = (mask.astype(np.float32) * 255).astype(np.uint8)
                    hue_hist = hue_analytic(cv2.imread(cube_image_file), mask, config)
                    cv2.imwrite(os.path.join(save_mask_dir, f"{k}_{category}.png"), mask_gray)
                    plot_hue(hue_hist, os.path.join(config.plot_dir, f"{basename}_{k}_hue.png"))
                    hue_sats = []
                    for hi in range(len(hue_hist)):
                        saturation_hist = saturation_analytic(cv2.imread(cube_image_file), mask,hi, config)
                        saturation_hist[0] *= 0.8
                        hue_sats.append(get_expect(saturation_hist))
                        plot_saturation(saturation_hist,
                                        config.colors[hi],
                                        config.saturations,
                                        os.path.join(config.plot_dir, f"{basename}_{k}_{config.colors[hi]}_sat.png"))
                    plot_hue_and_sat(hue_hist, hue_sats, os.path.join(config.plot_dir, f"{basename}_{k}_hue&sat.png"))

    print("CityColor runner end")
    return


if __name__ == '__main__':
    main()