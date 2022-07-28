import matplotlib.pyplot as plt
import json
import numpy as np


def read_data(path, file):
    with open(path + file, 'r') as f:
        pos_distance = json.loads(f.readline())
        neg_distance = json.loads(f.readline())
    return (pos_distance, neg_distance)


def single_plot(pos_distance):
    plt.hist(pos_distance, bins=5, histtype='bar', alpha=0.5)
    plt.legend(loc='best')
    plt.show()


def thresholds_plots(pos_distance, neg_distance):
    plt.hist([pos_distance, neg_distance], bins=3, histtype='bar', alpha=0.5, label=['positive', 'negative'])
    plt.show()


def stacked_plot(pos_distance, neg_distance, category):
    plt.figure(figsize=(20, 10), dpi=80)
    bins = np.arange(min(min(pos_distance), min(neg_distance)), max(max(pos_distance), max(neg_distance)), 0.01)

    # 男性乘客年龄直方图
    plt.hist(pos_distance, bins=bins, label='POS', edgecolor='k', color='steelblue', alpha=0.7)

    # 女性乘客年龄直方图
    plt.hist(neg_distance, bins=bins, label='NEG', edgecolor='k', alpha=0.5, color='r')

    # 调整刻度
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # 设置坐标轴标签和标题
    plt.title('the distance distribution, ' + category + ' as the unknown', fontsize=18)
    plt.xlabel('distance', fontsize=20)
    plt.ylabel('counts', fontsize=20)

    # 去除图形顶部边界和右边界的刻度
    plt.tick_params(top='off', right='off')

    # 显示图例
    plt.legend(loc='best', fontsize=20, labels=['POS', 'NEG'])

    # 显示图形
    plt.savefig(category + '_testing.pdf')

# path = '/Users/luyang/PycharmProjects/pythonProject/tripletLoss/'
# file = 'distance.txt'
#
# (pos_distance, neg_distance) = read_data(path, file)
#
#
# stacked_plot(pos_distance, neg_distance, 'training')
# thresholds_plots(pos_distance, neg_distance)
# single_plot(pos_distance)
# single_plot(neg_distance)
