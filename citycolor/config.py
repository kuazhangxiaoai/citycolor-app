class Config:
    def __init__(self):
        self.color_degree_num = 12 #色度
        self.saturation_degree_num = 5 #艳度
        self.value_degree_num = 10 #亮度

    @property
    def colors(self):
        return [i * (360 / self.color_degree_num) for i in range(self.color_degree_num)]

    @property
    def values(self):
        return [i * (1.0 / self.value_degree_num) for i in range(self.value_degree_num)]

    @property
    def saturations(self):
        return [i * (1.0 / self.saturation_degree_num) for i in range(self.saturation_degree_num)]