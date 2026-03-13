import cv2
from citycolor.config import Config
from ultralytics.utils import YAML
from ultralytics import YOLO
from ultralytics.utils.plotting import plot_predict_samples

config = Config()

def process_single(imagepath):
    img = cv2.imread(imagepath)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    try:
        model_path = "./weights/yolo11x-semseg.pt"
        data_path = "./ultralytics/cfg/datasets/CityscapesYOLO.yaml"
        model = YOLO(model_path)
        model.overrides["data"] = "./ultralytics/cfg/datasets/CityscapesYOLO.yaml"
        result = model(imagepath)[0]
        plot_predict_samples(
            result.orig_img,
            result.masks,
            nc=YAML.load(data_path)["nc"],
            colors=YAML.load(data_path)["colors"],
            fname="./assets/images/frankfurt.png",
            mname="./assets/masks/frankfurt.png",
            one_hot=True,
            overlap=False,
        )
    except Exception as e:
        print(e)

    #cv2.imshow("img", img_hsv)
    #cv2.waitKey()
    return

def main():
    print("CityColor runner start")
    img_path = "/media/yanggang/847C02507C023D84/CityEscape-YOLO/train/image/aachen_000000_000019_leftImg8bit__1024__0__10.png"
    process_single(img_path)
    print("CityColor runner end")
    return


if __name__ == '__main__':
    main()