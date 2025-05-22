# 导入pandas模块
import pandas as pd

# 读取CSV文件
cvsDataframe = pd.read_csv('/root/DeepClustermyproject/std_data/std.csv')


cvsDataframe.to_excel('/root/DeepClustermyproject/std_data/std.xlsx', index=None, header=True)