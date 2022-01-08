def get_index(my_list, num):
    # 获取list中值为num的所有index
    return [i for i, x in enumerate(my_list) if x == num]


def get_same_index_value(v, arr, targetArr):
    index_list = get_index(arr, v)
    v_list = []
    for i in index_list:
        v_list.append(targetArr[i])
    return v_list


# 遍历pp边
def handle_pp(p_init, top_pp_p1, top_pp_p2):
    pp_res = []
    pp_res.append(p_init)
    p_init1 = p_init
    p_init2 = p_init
    while 1:
        if p_init1 in top_pp_p1:
            index = top_pp_p1.index(p_init1)
            pp_res.append(top_pp_p2[index])
            p_init1 = top_pp_p2[index]
        else:
            break
    while 1:
        if p_init2 in top_pp_p2:
            index = top_pp_p2.index(p_init2)
            pp_res.append(top_pp_p1[index])
            p_init2 = top_pp_p1[index]
        else:
            break
    # p点去重
    return list(set(pp_res))


def migrate(top_pa_list_a, top_pa_list_p, top_a, top_p, top_pp_p1, top_pp_p2, top_vp_list_v, top_vp_list_p, top_v):
    ways = []
    # 排除坏数据
    for i in top_a:
        if i > 7955:
            break
        if i % 80 == 0:
            print('初始化训练数据-----------', i / 8000, '%')
        this_ways = []
        this_ways.append(i)
        # ap边p的节点
        pa_p_v = get_same_index_value(i, top_pa_list_a, top_pa_list_p)
        # v的值
        vp_v_v = get_same_index_value(pa_p_v[0], top_vp_list_p, top_vp_list_v)
        # pp边p2的节点
        # 找出所有与当前p相连的p边
        pp = handle_pp(pa_p_v[0], top_pp_p1, top_pp_p2)
        # 遍历 p=>a=>p
        for p in pp:
            this_ways.append(p)
            pa_a_v = get_same_index_value(p, top_pa_list_p, top_pa_list_a)
            for a_v in pa_a_v:
                # 将a的节点入list
                this_ways.append(a_v)
        # 将v加入list
        this_ways.append(vp_v_v[0])
        if (len(this_ways) != 257):
            print('i===', i)
            print('this_ways  len ==', len(this_ways))
        ways.append(this_ways)
    return ways
