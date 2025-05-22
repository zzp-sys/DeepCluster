import torch
from  torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
 
'''读取csv文件'''
def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=' ')#按行读取CSV文件中的数据,每一行以空格作为分隔符，再将内容保存成列表的形式
    next( plots)  # 读取首行
    x = []
    y = []
    for row in plots:
        x.append(float(row[0]))#从csv读取的数据是str类型，转成float类型
        y.append(float(row[1]))
    return x ,y
 
plt.figure()
x2,y2=readcsv("/root/DeepClustermyproject/std_data/std.csv")
plt.plot(x2, y2, color='red', linewidth = '1',label='std')
 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
 
plt.ylim(0.3, 0.5) #y轴取值范围
plt.xlim(0, 100) #x轴取值范围
plt.xlabel('Number of epochs',fontsize=15) #x轴标签
plt.ylabel('(1/d)^1/2',fontsize=15) #y轴标签
plt.legend(fontsize=16)  #标签的字号
plt.show()
plt.savefig("/root/DeepClustermyproject/std_data/std_fig")