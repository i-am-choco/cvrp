import numpy as np
import matplotlib.pyplot as plt

from cvrp import Cvrp
from ga import GA

if __name__ == '__main__':
    # 获取vrp问题基础数据
    cvrp = Cvrp()
    # GA参数
    ga = GA()
    # 最优成本
    bestCost = []
    # 最优路线
    bestVC = []
    # 计算各行向量之间的欧式距离
    distance = cvrp.get_distance(None)
    # 得出距离矩阵
    matrix = cvrp.get_distance_matrix(distance)
    # 更新染色体长度
    ga.update_choromsome_size(cvrp.customerLen + cvrp.vehicle - 1)
    # 初始化种群
    chromosome = ga.getinitialPopulation(ga.nind, ga.chromosome_size)
    """
        随机值初始解
    """
    currVC, nv, td, violate_num, violate_cus = ga.decode(chromosome[:, 0],cvrp.customerLen, ga.cap, cvrp.demands, matrix)
    curCost = ga.get_cost(currVC, matrix, cvrp.demands, ga.cap, ga.alpha, td)
    bestCost.append(curCost)
    print("初始化随机解: \n" + '车辆使用数目: ' + str(nv)  +  '，车辆行驶总距离：' +  str(td) + '，违反约束路径数目：' + str(violate_num) + '，违反约束顾客数目：' + str(violate_cus))
    print('\n成本：' + str(curCost))
    print('###')
    """
        ga循环
    """
    for i in range(0, 200):
        # 计算成本
        cost = ga.get_calObj(chromosome, cvrp.customerLen, ga.cap, cvrp.demands, matrix, ga.alpha)
        # 计算适应度
        fitness = ga.fitness(cost)
        # 选择操作
        selectCh = ga.select(chromosome, fitness, ga.generation_gap)
        # 交叉操作
        selectCh = ga.recombin(selectCh,ga.pc)
        # 变异操作
        selectCh = ga.mutate(selectCh, ga.pm)
        # 局部搜索
        selectCh = ga.localSearch(selectCh,cvrp.customerLen,ga.cap,cvrp.demands,matrix,ga.alpha)
        # 重插入子代的新种群
        chromosome = ga.reins(chromosome, selectCh, cost)
        # 删除种群中重复个体，并补齐删除的个体
        chromosome = ga.del_repeat(chromosome)
        # 输出最优解
        cost = ga.get_calObj(chromosome, cvrp.customerLen, ga.cap, cvrp.demands, matrix, ga.alpha)
        minCost = min(cost)
        bestCost.append(minCost)
        minIndex = np.where(cost == minCost)[0][0]
        currVC, nv, td, violate_num, violate_cus = ga.decode(chromosome[:, minIndex], cvrp.customerLen, ga.cap, cvrp.demands, matrix)
        bestVC = currVC
        print("第" + str(i+1) + "代最优解: \n" + '车辆使用数目: ' + str(nv) + '，车辆行驶总距离：' + str(td) + '，违反约束路径数目：' + str(
            violate_num) + '，违反约束顾客数目：' + str(violate_cus))
        print('\n成本：' + str(minCost))
        print('###')
    """
        结果可视化
    """
    cvrp.get_plot_route_map(bestVC)
    cvrp.get_plot_best_cost(bestCost)
    cvrp.get_plot_coordinates()
    plt.show()
