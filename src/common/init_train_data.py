import json
from src.utils.readExcel import read_excel
from src.common.init_edge import get_ways


def train_data():
    # lable格式
    y_data = []
    format_lable = [[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                    ]
    # 读数据
    print('开始读取节点数据')
    with open('ways.json', 'r') as f:
        data = json.load(f)
    # data = get_ways()
    # 读取源标签
    a_lable_path = "../../DataSets/a_lable.xls"
    a = read_excel(a_lable_path, 0)
    lable = read_excel(a_lable_path, 1)
    print('初始化训练集和标签')
    for i in lable:
        y_data.append(format_lable[int(i)])
    return data, y_data


