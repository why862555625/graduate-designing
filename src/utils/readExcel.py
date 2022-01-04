import xlrd
from src.utils.format import int_top

def read_excel(path, column):
    # 读取Excel数据
    workbook = xlrd.open_workbook(path)
    Data_sheet = workbook.sheets()[0]
    col_num0 = Data_sheet.col_values(column)
    col_num0 = int_top(col_num0)
    return col_num0
