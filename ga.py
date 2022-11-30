import numpy as np


class GA:

    def __init__(self):
        # 最大承重
        self.cap = 100
        # 违反的容量约束的惩罚函数系数
        self.alpha = 10
        # 种群大小
        self.nind = 50
        # 迭代次数
        self.maxgen = 200
        # 交叉概率
        self.pc = 0.9
        # 变异概率
        self.pm = 0.05
        # 代沟(Generation gap)
        self.generation_gap = 0.9
        # 染色体长度=顾客数目+车辆最多使用数目-1
        self.chromosome_size = 0
        # self.gla = ContinuousGenAlgSolver(
        #     n_genes=100,
        #     fitness_function=self.get_fitness,
        #     max_gen=self.maxgen,
        #     pop_size=self.nind,
        #     mutation_rate=self.pm,
        #     selection_rate=self.pc,
        #     variables_limits=(0, self.nind)
        # )
        # chrome = gla.initialize_population()
        # print(gla.calculate_fitness())

    def find(condition):
        res = np.nonzero(condition)
        return res

    # 更新染色体长度
    def update_choromsome_size(self, size):
        self.chromosome_size = size

    # 随机生成初始化种群
    def getinitialPopulation(self, length, populationSize):
        chromsomes = np.zeros((populationSize, length), dtype=np.int)
        for popusize in range(populationSize):
            # np.random.randit()产生[0,种群大小)之间的随机整数，第三个参数表示随机数的数量
            chromsomes[popusize, :] = np.random.randint(0, populationSize, length)
        return chromsomes

    # 整理vc空数据
    def del_vc(self, vc):
        fv = [i for i in vc if i != []]
        return fv

    def leave_load(self, route, demands):
        """
        :param route: 一条配送路线
        :param demands: 表示由配送中心运送到顾客的配送量
        :return: 计算某一条路径上离开配送中心时的载货量
        """
        # 配送路线经过顾客的总数量
        n = len(route)
        # 初始车辆在配送中心时的装货量为0
        ld = 0
        if n != 0:
            for i in range(0, n):
                if route[i] != 0:
                    ld = ld + demands[route[i]]
        return ld

    # 判断一条路线是否满足载重量约束，1表示满足，0表示不满足
    def judge_route(self, route, demands, cap):
        """
        :param route: 路线
        :param demands: 顾客需求量
        :param cap: 车辆最大装载量
        :return: 标记一条路线是否满足载重量约束，1表示满足，0表示不满足
        """
        # 计算该条路径上离开配送中心时的载货量
        ld = self.leave_load(route, demands)
        if ld > cap:
            return 0
        else:
            return 1

    def part_length(self, route, dist):
        """
        计算一条路线的路径长度
        :param route: 路径
        :param dist: 距离矩阵
        :return: 计算一条路线的路径长度
        """
        # 当前路径长度
        n = len(route)
        # 累计总长度
        pl = 0
        if n != 0:
            for i in range(0, n):
                if i == 0:
                    pl = pl + dist[0, route[i]]
                else:
                    pl = pl + dist[route[i - 1], route[i]]
            pl = pl + dist[route[-1], 0]
        return pl

    def travel_distance(self, vc, dist):
        """
        计算每辆车所行驶的距离，以及所有车行驶的总距离
        :param vc: 每辆车所经过的顾客
        :param dist: 距离矩阵
        :return sumTD: 所有车行驶的总距离
        :return everyTD: 每辆车所行驶的距离
        """
        # 车辆数
        n = len(vc)
        everyTD = np.zeros(n)
        for i in range(0, n):
            # 每辆车所经过的顾客
            part = vc[i]
            if len(part) != 0:
                everyTD[i] = self.part_length(part, dist)
        sumTD = sum(everyTD)
        return sumTD

    # 解码
    def decode(self, chrom, cusnum, cap, demands, dist):
        """
        :param chrom: 个体
        :param cusnum: 顾客数目
        :param cap: 最大载重量
        :param demands: 需求量
        :param dist: 距离矩阵，满足三角关系，暂用距离表示花费c[i][j]=dist[i][j]
        :return vc: 每辆车所经过的顾客，是一个cell数组
        :return nv: 车辆使用数目
        :return td: 车辆行驶总距离
        :return violate_num: 违反约束路径数目
        :return violate_cus: 违反约束顾客数目
        """
        violate_num = 0
        violate_cus = 0

        # 车辆计数器，表示当前车辆使用数目
        count = 0
        # 找出个体中配送中心的位置
        location0 = np.where(chrom > cusnum)[0]
        # 每辆车所经过的顾客
        vc = [[]] * (len(location0) + 1)
        # 过滤配送中心位置
        for i in range(0, len(location0)):
            route = None
            # 第一个配送中心的位置
            if i == 0:
                index = location0[i]
                # 提取两个配送中心间的路径
                route = chrom[0:index + 1]
                # 删除路径中配送中心序号
                route = np.delete(route, -1)
            else:
                pIndex = location0[i - 1] + 1
                curIndex = location0[i]
                # 提取两个配送中心之间的路径
                route = chrom[pIndex: curIndex + 1]
                # 删除路径中配送中心序号
                route = np.delete(route, -1)

            # 更新配送方案
            vc[count] = route
            # 车辆使用数目
            count = count + 1
        # 最后一条路径
        route = chrom[location0[-1] + 1:]
        vc[count] = route
        # 将vc中空的数组移除
        vc = self.del_vc(vc)
        # 所使用的车辆数
        nv = len(vc)
        for i in range(0, nv):
            curRoute = vc[i]
            # 判断一条路线是否满足载重量约束，1表示满足，0表示不满足
            flag = self.judge_route(curRoute, demands, cap)
            if flag == 0:
                # 如果这条路径不满足约束，则违反约束顾客数目加该条路径顾客数目
                violate_cus = violate_cus + len(curRoute)
                # 如果这条路径不满足约束，则违反约束路径数目加1
                violate_num = violate_num + 1
        # 该方案车辆行驶总距离
        td = self.travel_distance(vc, dist)
        return vc, nv, td, violate_num, violate_cus

    def violateLoad(self, vc, demands, cap):
        """

        :param vc: 每辆车所经过的顾客
        :param demands: 各个顾客需求量
        :param cap: 车辆最大载货量
        :return: 计算当前解违反的容量约束
        """
        # 所用车辆数量
        nv = len(vc)
        q = 0
        for i in range(0, nv):
            route = vc[i]
            ld = self.leave_load(route, demands)
            if ld > cap:
                q = q + ld - cap
        return q

    def get_cost(self, vc, dist, demands, cap, alpha, td):
        """
        计算当前解的成本函数
        :param vc: 每辆车所经过的顾客
        :param dist: 距离矩阵
        :param demands: 各个顾客需求量
        :param cap: 车辆最大载货量
        :param alpha: 违反的容量约束的惩罚函数系数
        :param td: 车辆行驶总距离
        :return:  f=TD+alpha*q
        """
        q = self.violateLoad(vc, demands, cap)
        cost = td + alpha * q
        return cost

    # 计算种群的目标函数值
    def get_calObj(self, chrom, cusnum, cap, demands, dist, alpha):
        """
        :param chrom: 种群
        :param cusnum: 顾客数目
        :param cap: 最大载重量
        :param demands: 需求量
        :param dist: 距离矩阵，满足三角关系，暂用距离表示花费c[i][j] = dist[i][j]
        :param alpha: 违反的容量约束的惩罚函数系数
        :return: 每个个体的目标函数值，定义为车辆使用数目*10000+车辆行驶总距离
        """
        print('计算种群的目标函数值')
        # 种群数目
        nind = len(chrom[0])
        # 储存每个个体函数值
        objV = np.zeros(nind)
        for i in range(0, nind):
            # 取列数据
            vc, nv, td, violate_num, violate_cus = self.decode(chrom[:, i], cusnum, cap, demands, dist)
            cost = self.get_cost(vc, dist, demands, cap, alpha, td)
            objV[i] = cost
        return objV

    def fitness(self, objV):
        """
        适配值函数
        :param objV: 个体的长度
        :return: 个体的适应度值
        """
        n = len(objV)
        fitness = np.zeros(n)
        for i in range(0, n):
            fitness[i] = 1. / objV[i]
        return fitness
