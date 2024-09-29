import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n = 10000
r = 10  # 空桌子假象人数

class cluster:
    count = 0
    clusters = []
    index = 0

    def __init__(self, index):
        self.index = index

    def __str__(self):
        return "Table " + str(self.index) + " : " + str(self.count)

    def add(self, i):
        self.count += 1
        self.clusters.append(i)
        return self

if __name__ == "__main__":
    first_table = cluster(0).add(0)  # 第一位顾客选择第一个桌子
    restaurant = [first_table]

    for i in range(1, n):
        customerSumWithEmpty = i + r
        p = [] # 构造概率分布
        for table in restaurant:
            p.append(table.count / customerSumWithEmpty)
        p.append(r / customerSumWithEmpty)  # 空桌子号码设为最后一个数字
        choice = np.random.choice(len(p), 1, p=p) # 从len(p)范围中随机选取一个元素，选取时的概率分布为p
        t = choice[0]  # p = p
        if t == len(restaurant):  # 空桌子
            restaurant.append(cluster(t))
        restaurant[t].add(i)

        if (i + 1) % 10 == 0:
            print(f"after {i + 1} customers：", end='\t')
            for table in restaurant:
                print(table.count, end=' ')
            print(' ')