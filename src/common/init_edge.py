from src.utils.readExcel import read_excel
from src.utils.format import get_index, ergodic_pa
import json

# linux path
# 获取数据集
# a_path = "/home/wanghaiyun/桌面/SN/SN/DataSets/DBLP/author.xlsx"
# pa_path = "/home/wanghaiyun/桌面/SN/SN/DataSets/DBLP/pa边.xlsx"
# pp_path = "/home/wanghaiyun/桌面/SN/SN/DataSets/DBLP/pp边.xlsx"
# pv_path = "/home/wanghaiyun/桌面/SN/SN/DataSets/DBLP/pv边.xlsx"
# 所有a点
# windows

a_path = "../../DataSets/DBLP/author.xlsx"
pa_path = "../../DataSets/DBLP/pa边.xlsx"
pp_path = "../../DataSets/DBLP/pp边.xlsx"
pv_path = "../../DataSets/DBLP/pv边.xlsx"

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
# pv边 v顶点
top_pv_v = read_excel(pv_path, 3)
# pv边 p顶点数量
top_num_pv_p = read_excel(pv_path, 1)
# pv边 v顶点数量
top_num_pv_v = read_excel(pv_path, 3)
# pv边 v顶点去重
top_v = list(set(top_pv_v))
# pp
# pv边 第一个p顶点
top_pp_p1 = read_excel(pp_path, 0)
# pv边 第一个p顶点数量
top_num_pp_p1 = read_excel(pp_path, 1)
# pv边 第二个p顶点
top_pp_p2 = read_excel(pp_path, 3)
# pv边 第二个p顶点数量
top_num_pp_p2 = read_excel(pp_path, 4)
# 第一个p顶点去重
top_p = list(set(top_pa_p))

p = {}
# 整理数据结构 p={'index':{"a":[],"p":[],"v":[]}}

for i in top_p:
    # 遍历top_p 获取p==i的所有下标
    index_pa = get_index(top_pa_p, i)

    for pa in index_pa:
        # 看是否已经有属性了
        if i in p:
            # 有效数据长度 后边有一些坏数据直接舍弃
            if top_pa_a[pa] < 10153:
                # {"a": [], "p": [], "v": []}
                # p[index]["a"] =[[index,边权重]]
                edge_num_pa = 2 / ((1 / top_num_pa_p[pa] + 1))
                p[i]["a"].append([top_pa_a[pa], round(edge_num_pa, 2)])
        else:
            p[top_pa_p[pa]] = {"a": [], "p": [], "v": []}
            edge_num_pa = 2 / ((1 / top_num_pa_p[pa] + 1))
            p[i]["a"].append([top_pa_a[pa], round(edge_num_pa, 2)])
    # PP边
    index_pp = get_index(top_pp_p1, i)
    for pp in index_pp:
        # edge_num_pp = 2 / (1 / top_num_pp_p1[pp] + 1 / top_num_pp_p2[pp])
        edge_num_pp = 1
        if top_pp_p2[pp] < 10153:
            p[i]["p"].append([top_pp_p2[pp], edge_num_pp])
    index_pv = get_index(top_pv_p, i)
    for pv in index_pv:
        # 权重 pp计算公式
        edge_num_pv = 2 / (1 / top_num_pv_p[pv] + 1 / top_num_pv_v[pv])
        p[i]["v"].append([top_pp_p2[pv], edge_num_pv])

# 整理数据结构 a={"a":[p,num]}
a = {}
for i in top_a:
    index_ap = get_index(top_pa_a, i)
    for j in index_ap:
        edge_num_ap = 2 / ((1 / top_num_pa_p[j] + 1))
        if i in a:
            a[i].append([top_pa_p[j], round(edge_num_ap, 2)])
        else:
            a[i] = []
            a[i].append([top_pa_p[j], round(edge_num_ap, 2)])

# 初始化V的结构
v = {}
for i in top_v:
    index_vp = get_index(top_pv_v, i)
    for j in index_vp:
        edge_num_vp = 2 / ((1 / top_num_pv_v[j] + 1 / top_num_pv_p[j]))
        if i in v:
            v[i].append([top_pv_p[j], edge_num_vp])
        else:
            v[i] = []
            v[i].append([top_pv_p[j], edge_num_vp])

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
