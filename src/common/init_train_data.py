import json
from src.utils.readExcel import read_excel
from src.common.init_edge import get_ways


def train_data():
    # lable格式
    # 将标签转换为热编码
    loop_index = 0
    y_data = []
    # 原版的标签
    y_data_original = []
    format_lable = [[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                    ]
    # 读数据
    print('开始读取节点数据')
    with open('../main/ways.json', 'r') as f:
        data = json.load(f)
    # data = get_ways()
    # 读取源标签
    a_lable_path = "../../DataSets/a_lable.xls"
    # a标签
    a = read_excel(a_lable_path, 0)
    # a的标签
    lable = read_excel(a_lable_path, 1)
    print('初始化训练集和标签')
    for i in lable:
        if loop_index > 7955:
            break
        y_data_original.append(int(i))
        y_data.append(format_lable[int(i)])
        loop_index += 1
    return data, y_data, y_data_original
