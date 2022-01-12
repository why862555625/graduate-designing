def get_index(my_list, num):
    # 获取list中值为num的所有index
    return [i for i, x in enumerate(my_list) if x == num]


def get_same_index_value(v, arr, targetArr):
    # 获取第一个list中值为v的index
    index_list = get_index(arr, v)
    v_list = []
    for i in index_list:
        # 返回第二个list相同index的值   也就是两个节点有边的情况
        v_list.append(targetArr[i])
    return v_list


# 遍历pp边
def handle_pp(p_init, top_pp_p1, top_pp_p2):
    pp_res = []
    # 将第一个p节点放入list
    pp_res.append(p_init)
    p_init1 = p_init
    p_init2 = p_init
    # 往前遍历当前p节点可以到达的其他p节点
    while 1:
        # 如果当前p节点在top_pp_p1中说明可以往前遍历到
        if p_init1 in top_pp_p1:
            index = top_pp_p1.index(p_init1)
            pp_res.append(top_pp_p2[index])
            # 切换当前p节点为到达的p节点  继续向前遍历
            p_init1 = top_pp_p2[index]
        else:
            # 走到头了  往前遍历已经没有可以到达的p节点了
            break
    # 往后遍历当前p节点可以到达的其他p节点
    while 1:
        if p_init2 in top_pp_p2:
            index = top_pp_p2.index(p_init2)
            pp_res.append(top_pp_p1[index])
            # 切换当前p节点为到达的p节点  继续向后遍历
            p_init2 = top_pp_p1[index]
        else:
            # 走到头了  往后遍历已经没有可以到达的p节点了
            break
    # 将所有当前p节点可以到达的p节点返回
    return list(set(pp_res))


def migrate(top_pa_list_a, top_pa_list_p, top_a, top_p, top_pp_p1, top_pp_p2, top_vp_list_v, top_vp_list_p, top_v):
    ways = []
    for i in top_a:
        # a节点值范围
        if i > 7955:
            break
        # 初始化进程可视化
        if i % 80 == 0:
            print('初始化训练数据-----------', i / 8000, '%')
        # 从当前a出发遍历的节点
        this_ways = []
        # 将 当前a节点加入list
        this_ways.append(i)
        # 与当前a节点相连的p节点
        pa_p_v = get_same_index_value(i, top_pa_list_a, top_pa_list_p)
        # 当前a节点下v节点的值
        vp_v_v = get_same_index_value(pa_p_v[0], top_vp_list_p, top_vp_list_v)
        # pp边p2的节点
        # 找出所有与当前p相连的p边
        pp = handle_pp(pa_p_v[0], top_pp_p1, top_pp_p2)
        # 遍历 p=>a=>p=>a.....
        for p in pp:
            # 将p入list
            this_ways.append(p)
            # 找到与当前p相连的所有a
            pa_a_v = get_same_index_value(p, top_pa_list_p, top_pa_list_a)
            for a_v in pa_a_v:
                # 将a的节点入list
                this_ways.append(a_v)
        # 将v加入list
        this_ways.append(vp_v_v[0])
        if (len(this_ways) != 257):
            print('i===', i)
            print('this_ways  len ==', len(this_ways))
        # 将当前a遍历的路径加入list
        ways.append(this_ways)
    # 返回训练集
    return ways
