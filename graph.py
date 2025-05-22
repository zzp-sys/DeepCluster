import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_excel("/root/DeepClustermyproject/different_dataset/cifar10.xlsx")
x = df["Column1.1"]
y1 = df["Column1.2"]
y2 = df["Column1.3"]
y3 = df["Column1.4"]

plt.figure(figsize=(8, 6))  # 可选，设置图形的尺寸

plt.plot(x, y1, label="NMI")
plt.plot(x, y2, label="ACC")
plt.plot(x, y3, label="ARI")

plt.ylim(0, 1)

plt.gca().xaxis.tick_bottom()
plt.gca().yaxis.tick_left()

plt.xlabel("Number of epochs")
plt.ylabel("clustering")
plt.title("cifar10")
plt.legend()  # 添加图例


plt.show()
plt.savefig("/root/DeepClustermyproject/different_dataset/cifar10.svg", bbox_inches='tight')
