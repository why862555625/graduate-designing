import json
from src.utils.readExcel import read_excel
from src.common.init_edge import get_ways


def train_data():
    # lable格式
    y_data = []
    lable = {"0": [1, 0, 0, 0, 0],
             "1": [0, 1, 0, 0, 0],
             "2": [0, 0, 1, 0, 0],
             "3": [0, 0, 0, 1, 0],
             "4": [0, 0, 0, 0, 1],
             }
    # 读数据
    # with open('ways.json', 'r') as f:
    #     data = json.load(f)
    data = get_ways()
    # 读取源标签
    a_lable_path = "../../DataSets/a_lable.xls"
    a = read_excel(a_lable_path, 0)
    lable = read_excel(a_lable_path, 1)
    for i in data:
        num = 0
        for j in i:
            if j < 8000:
                num = j
        this_lable = lable[num - 1]
        y_data.append(this_lable)
    return data, y_data

train_data()
