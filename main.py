import pandas as pd

from cvrp import Cvrp
from ga import GA

if __name__ == '__main__':
    # 获取vrp问题基础数据
    cvrp = Cvrp()
    # GA参数
    ga = GA()
    # 计算各行向量之间的欧式距离
    distance = cvrp.get_distance(None)
    # 得出距离矩阵
    matrix = cvrp.get_distance_matrix(distance)
    # 更新染色体长度
    ga.update_choromsome_size(cvrp.customerLen + cvrp.vehicle - 1)
    # 初始化种群
    chromosome = ga.getinitialPopulation(ga.nind, ga.chromosome_size)
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