import copy


def get_index(my_list, num):
    # 获取list中值为num的所有index
    return [i for i, x in enumerate(my_list) if x == num]


def int_top(top_list):
    # 格式化 全部转化为int类型
    top_list = list(map(int, top_list))
    return top_list


def max_lens(m):
    # 求所有列表中的列表 最长的列表的长度
    max_len = 0
    for i in m:
        if len(m[i]) > max_len:
            max_len = len(m)
    return max_len


def ergodic_pa(now, p):
    ways = []
    # 先遍历一次
    # 深拷贝 避免影响原数据
    copy_p = copy.deepcopy(p)
    # 如果p中还有没有遍历完的a
    while len(copy_p[now]["a"]) > 0:
        # 将 a点加入
        ways.append(copy_p[now]["a"].pop(0)[0])
        # 将p点加入
        ways.append(now)
    # 继续用最新的
    copy_p = copy.deepcopy(p)
    # 遍历其中的p点
    if len(copy_p[now]["p"]) > 0:
        # 看有多少个p
        lens = []
        for i in range(0, len(copy_p[now]["p"]) - 1):
            lens.append(i)
        # 从大到小排序
        lens = reversed(lens)
        for i in lens:
            #
            if len(copy_p[now]["p"]) > 0:
                ways.extend(ergodic_pa(copy_p[now]["p"].pop(i)[0], copy_p))
            copy_p = copy.deepcopy(p)
    return ways
