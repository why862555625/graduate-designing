from src.utils.readExcel import read_excel

pa_path = "../../DataSets/pa.xls"
top_pa_p = read_excel(pa_path, 0)
print(top_pa_p)
