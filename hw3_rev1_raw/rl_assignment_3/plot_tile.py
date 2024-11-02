import matplotlib.pyplot as plt

# 数据
tiles = ['16','32','64', '128', '256', '512', '1024']
counts = [1,0,10,52, 345, 529, 63]

# 创建直方图
plt.figure(figsize=(10, 6))
plt.bar(tiles, counts)

# 设置标题和标签
plt.title('Best tile distribution', fontsize=16)
plt.xlabel('Best tile', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# 在每个柱子上方添加数值标签
for i, v in enumerate(counts):
    plt.text(i, v, str(v), ha='center', va='bottom')

# 显示图形
plt.show()