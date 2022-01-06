from src.utils.readExcel import read_excel
from src.utils.format import migrate
import json

pa_path = "../../DataSets/pa.xls"
pp_path = "../../DataSets/pp.xls"
vp_path = "../../DataSets/vp.xls"

# pa
top_pa_list_a = read_excel(pa_path, 1)
top_pa_list_p = read_excel(pa_path, 0)
top_a = list(set(top_pa_list_a))
top_p = list(set(top_pa_list_p))
# pp
top_pp_p1 = read_excel(pp_path, 0)
top_pp_p2 = read_excel(pp_path, 1)

# vp
top_vp_list_v = read_excel(vp_path, 0)
top_vp_list_p = read_excel(vp_path, 1)
top_v = list(set(top_vp_list_v))


# 开始遍历
def get_ways():
    res_data = migrate(top_v, top_vp_list_v, top_vp_list_p, top_pa_list_p, top_pa_list_a)
    with open('ways.json', 'w') as f:
        json.dump(res_data, f)

get_ways()
# with open('ways.json', 'w') as f:
#     json.dump(ways, f)

# Reading data back
# with open('ways.json', 'r') as f:
#     data = json.load(f)
# print(ways)
