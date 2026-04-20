import os

class Config:
    def __init__(self, project_dir):
        self.cube_dir = os.path.join(project_dir,  "assets/cube")
        self.img_dir = os.path.join(project_dir, "assets/panos")
        self.save_image_dir = os.path.join(project_dir, "assets/images")
        self.save_mask_dir = os.path.join(project_dir, "assets/masks")
        self.categories_file = os.path.join(project_dir, "ultralytics/cfg/datasets/CityscapesYOLO.yaml")
        self.plot_dir = os.path.join(project_dir, "assets/plot")
        self.color_system_file = os.path.join(project_dir, "assets/bj-color-system.json")
        self.attention_direction = ['px', 'nx']
        self.attention_categories = ['building'] #需要查询ultralytics的cityscapesYOLO配置文件的对照表

        self.color_degree_num = 12 #色度
        self.saturation_degree_num = 5 #艳度
        self.value_degree_num = 10 #亮度

    @property
    def colors(self):
        return [i * (360 / self.color_degree_num) for i in range(self.color_degree_num)]

    @property
    def color_resolution(self):
        return 360 / self.color_degree_num

    @property
    def values(self):
        """
        计算亮度
        :param s: 艳度等级
        :return:
        """
        return [i * (1.0 / self.value_degree_num) for i in range(self.value_degree_num)]

    @property
    def value_resolution(self):
        return 1.0 / self.value_degree_num

    @property
    def saturations(self):
        return [i * (1.0 / self.saturation_degree_num) for i in range(self.saturation_degree_num)]

    @property
    def saturation_resolution(self):
        return 1.0 / self.saturation_degree_num
