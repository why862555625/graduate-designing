import xlrd


# 路径    Excel中的第几列
def read_excel(path, column):
    # 读取Excel数据
    workbook = xlrd.open_workbook(path)
    Data_sheet = workbook.sheets()[0]
    col_num0 = Data_sheet.col_values(column)
    return col_num0
