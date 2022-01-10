from src.utils.readExcel import read_excel
from src.utils.format import migrate
import json

pa_path = "../../DataSets/pa.xls"
pp_path = "../../DataSets/pp.xls"
vp_path = "../../DataSets/vp.xls"
# 从Excel中读数据
# pa边
top_pa_list_a = read_excel(pa_path, 1)
top_pa_list_p = read_excel(pa_path, 0)
top_a = list(set(top_pa_list_a))
top_p = list(set(top_pa_list_p))
# pp边
top_pp_p1 = read_excel(pp_path, 0)
top_pp_p2 = read_excel(pp_path, 1)
# vp边
top_vp_list_v = read_excel(vp_path, 0)
top_vp_list_p = read_excel(vp_path, 1)
top_v = list(set(top_vp_list_v))


# 开始遍历
def get_ways():
    res_data = migrate(top_pa_list_a, top_pa_list_p, top_a, top_p, top_pp_p1, top_pp_p2, top_vp_list_v, top_vp_list_p,
                       top_v)
    with open('ways.json', 'w') as f:
        json.dump(res_data, f)
    return res_data
