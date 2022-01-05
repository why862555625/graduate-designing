def get_index(my_list, num):
    # 获取list中值为num的所有index
    return [i for i, x in enumerate(my_list) if x == num]


def migrate(top_v, top_vp_list_v, top_vp_list_p, top_pa_list_p, top_pa_list_a):
    ways = []
    index = 0
    for i in top_v:
        ways.append([i])
        index_vp_p_list = get_index(top_vp_list_v, i)
        # 这次循环中出现的p点
        ways_p_list = []
        # 遍历所有与v有边的p
        for j in index_vp_p_list:
            # 将所有的p加入
            ways_p_list.append(top_vp_list_p[j])
        ways[index].extend(ways_p_list)
        # 遍历所有的pa边a
        for n in ways_p_list:
            index_pa_p_list = get_index(top_pa_list_p, n)
            for m in index_pa_p_list:
                index_pa_a_list = get_index(top_pa_list_a, m)
                ways[index].extend(index_pa_a_list)
        index += 1
    return ways


