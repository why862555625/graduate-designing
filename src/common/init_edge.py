from src.utils.readExcel import read_excel
from src.utils.format import get_index, ergodic_pa
import json

pa_path = "../../DataSets/pa.xls"
pp_path = "../../DataSets/pp.xls"
pv_path = "../../DataSets/pv.xls"

# pa
# pa边 p顶点
top_pa_p = read_excel(pa_path, 0)
# pa边 p顶点的数量
top_num_pa_p = read_excel(pa_path, 1)
# pa边 a顶点
top_pa_a = read_excel(pa_path, 3)
# pa边 a顶点去重    （主要是第一个顶点会重复）
top_a = list(set(top_pa_a))

# pv
# pv边 p顶点
top_pv_p = read_excel(pv_path, 0)


# 开始遍历
ways = {}
a_len = len(top_a)
a_max = max(top_a)
# 从a点出发 a=>p=>p=>v
for i in top_a:
    if i % 100 == 0:
        print(round(i * 100 / a_max, 2), "%")
    ways[i] = []
    # 将起始点加入ways
    ways[i].append(i)
    # 将第一个p点加入ways
    ways[i].append(a[i][0][0])
    # 开始将遍历点加入
    ways[i].extend(ergodic_pa(a[i][0][0], p))

print(ways)
with open('ways.json', 'w') as f:
    json.dump(ways, f)

# Reading data back
# with open('ways.json', 'r') as f:
#     data = json.load(f)
# print(ways)
