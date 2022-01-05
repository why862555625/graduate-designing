import json
from src.utils.readExcel import read_excel
from src.utils.format import get_index, max_lens


def train_data():
    # lable格式
    lable = {"0": [1, 0, 0, 0],
             "1": [0, 1, 0, 0],
             "2": [0, 0, 1, 0],
             "3": [0, 0, 0, 1],
             }
    # 读数据
    with open('ways.json', 'r') as f:
        data = json.load(f)
    # 读取源标签
    path = "../../DataSets/DBLP/a标签.xltx"
    # 求路径中最长的lens
    max_len = max_lens(data)
    print("max_len=========", max_len)

    a = read_excel(path, 0)
    y = read_excel(path, 1)
    x_data = []
    y_data = []
    for i in data:
        index_i = get_index(a, i)
        while (len(data[i]) < max_len):
            # 不足最大长度的以 -1 补充
            data[i].append(-1)
        # 数据
        x_data.append(data[i])
        # 数据标签
        y_data.append(lable[y[index_i[0]]])
    # 将数据集返回
    return x_data, y_data


x_data, y_data = train_data()
