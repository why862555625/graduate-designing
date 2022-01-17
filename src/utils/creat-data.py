import xlwt


def save_excel(target_list, output_file_name):
    """
    将数据写入xls文件
    """
    if not output_file_name.endswith('.xls'):
        output_file_name += '.xls'
    workbook = xlwt.Workbook(encoding='utf-8')
    ws = workbook.add_sheet("sheet1")

    for i in range(len(target_list)):
        for j in range(2):
            ws.write(i, j, target_list[i][j])
    workbook.save(output_file_name)


def vp():
    data = []
    j = 8001
    for i in range(10001, 10041):
        for _ in range(50):
            data.append([i, j])
            j += 1
    return data


def pa():
    j = 0
    result = []
    for i in range(8001, 10001):
        for _ in range(4):
            result.append([i, j])
            j += 1
    return result


def pp():
    m = 8001
    n = 8002
    res = []
    while (n <= 10000):
        for _ in range(50):
            res.append([m, n])
            m += 1
            n += 1
        m += 1
        n += 1
    return res


def a_lable():
    lable = []
    x = -1
    for i in range(8001):
        if (i % 2000 == 0):
            if (i != 8000):
                x += 1
        lable.append([i, x])
    return lable


vp_data = vp()
vp_path = "../../DataSets/DBLP/vp.xls"
save_excel(vp_data, vp_path)
pp_data = pp()
pp_path = "../../DataSets/DBLP/pp.xls"
save_excel(pp_data, pp_path)
pa_data = pa()
pa_path = "../../DataSets/DBLP/pa.xls"
print(pa_data)
save_excel(pa_data, pa_path)
a_lable_data = a_lable()
a_lable_path = "../../DataSets/DBLP/a_lable.xls"
save_excel(a_lable_data, a_lable_path)
