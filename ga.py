import math

import numpy as np
import pandas as pd


class GA:

    def __init__(self):
        # 最大承重
        self.cap = 100
        # 违反的容量约束的惩罚函数系数
        self.alpha = 10
        # 种群大小
        self.nind = 50
        # 迭代次数
        self.maxgen = 400
        # 交叉概率
        self.pc = 0.7
        # 变异概率
        self.pm = 0.02
        # 代沟(Generation gap)
        self.generation_gap = 0.9
        # 染色体长度=顾客数目+车辆最多使用数目-1
        self.chromosome_size = 0

    # 更新染色体长度
    def update_choromsome_size(self, size):
        self.chromosome_size = size

    # 随机生成初始化种群
    def getinitialPopulation(self, length, populationSize):
        chromsomes = np.zeros((populationSize, length), dtype=np.int_)
        newC = np.zeros((length, populationSize), dtype=int)
        for popusize in range(populationSize):
            # np.random.randit()产生[0,种群大小)之间的随机整数，第三个参数表示随机数的数量
            chromsomes[popusize, :] = np.random.randint(0, populationSize, length)
        for popusize in range(length):
            # np.random.randit()产生[0,种群大小)之间的随机整数，第三个参数表示随机数的数量
            newC[popusize, :] = np.random.choice(populationSize, populationSize, replace=False)
        return newC.transpose()

    # 整理vc空数据
    def del_vc(self, vc):
        fv = [i for i in vc if i.size > 0]
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

    def sus(self, fitness, nsel):
        """

            :param fitness: 个体的适应度值
            :param nsel: 被选择个体的数目
            :return: 被选择个体的索引号
            """
        nind = len(fitness)
        ans = np.ndim(fitness)
        # 执行随机常规采样
        cumfit = np.cumsum(fitness)
        # 平均适应度
        average = cumfit[-1] / nind
        # [0,1)随机数
        random = np.random.random()
        # nsel举证
        A = np.arange(start=0, stop=nsel, step=1)
        trials = average * (random + np.transpose(A))
        Mf = np.ones((nind, nsel))
        for i in range(0, nind):
            Mf[i] = cumfit[i]
        Mt = np.ones((nind, nsel))
        for i in range(0, nsel):
            Mt[::, i] = trials[i]
        Ma = np.zeros(nsel)
        Ma = np.insert(Mf[0:nind - 1, :], 0, Ma, axis=0)
        pMa = pd.DataFrame(Ma)
        pMf = pd.DataFrame(Mf)
        pMt = pd.DataFrame(Mt)
        pRf = pMt < pMf
        pRa = pMa <= pMt
        pRt = pd.DataFrame(pRf == pRa)
        NewChrIx = []
        ans = []
        shuf = []
        Rt = pRt.to_numpy()
        for i in range(0, len(Rt)):
            for j in range(0, len(Rt[i])):
                if Rt[i][j] == True:
                    NewChrIx.append(i)
                    ans.append(Mt[i, j])
        randMt = np.random.rand(nsel, 1)
        srandMt = np.sort(randMt, axis=0)
        for i in range(0, len(srandMt)):
            shuf.append(np.where(randMt == srandMt[i])[0][0])
        # 根据shuf的序号取值
        resultIndex = []
        for i in range(0, len(shuf)):
            resultIndex.append(NewChrIx[shuf[i]])
        return resultIndex

    def select(self, chrom, fitness, ggap):
        """
            选择操作
            :param chrom:种群
            :param fitness:适应度值
            :param ggap:选择概率
            :return: 被选择的个体
            """
        # 种群数量
        nind = len(chrom[0])
        # 被选择个体的数目
        nsel = max(math.floor(nind * ggap + 0.5), 2)
        # 被选择个体的索引号
        chrIndex = self.sus(fitness, nsel)
        chr = pd.DataFrame(chrom).transpose()
        newChrom = []
        for i in chrIndex:
            item = chr.query('index ==' + str(i))
            newChrom.append(pd.DataFrame(item).to_numpy()[0])
        return pd.DataFrame(newChrom).transpose().to_numpy()

    def ox(self, a, b):
        """

            :param a: 待交换个体
            :param b: 待交换个体
            :return: 交叉后得到的两个个体
            """
        length = len(a)
        resultA = []
        resultB = []
        while 1:
            r1 = np.random.randint(length)
            r2 = np.random.randint(length)

            if r1 is not r2:
                s = min(r1, r2)
                e = max(r1, r2)
                na = np.insert(a, 0, b[s:e + 1])
                nb = np.insert(b, 0, a[s:e + 1], axis=0)
                for i in range(0, len(na)):
                    aIndex = np.where(na == na[i])[0]
                    bIndex = np.where(nb == nb[i])[0]
                    if len(aIndex) > 1:
                        na[aIndex[1]] = -1
                    if len(bIndex) > 1:
                        nb[bIndex[1]] = -1
                    if (i + 1) == len(na):
                        break
                for i in range(0, len(na)):
                    if na[i] != -1:
                        resultA.append(na[i])
                for i in range(0, len(nb)):
                    if nb[i] != -1:
                        resultB.append(nb[i])
                break
        return resultA, resultB

    def recombin(self, chrom, pc):
        """
            OX交换操作
            :param chrom: 被选择个体
            :param pc: 交叉概率
            :return: 交叉后的个体
            """
        # 被选择个体的数目
        nsel = len(chrom[0])
        nchrom = pd.DataFrame(chrom).transpose().to_numpy()
        end = nsel - np.mod(nsel, 2)
        for i in range(0, end, 2):
            # [0,1)随机数
            random = np.random.random()
            if pc >= random:
                nchrom[i, :], nchrom[i + 1, :] = self.ox(nchrom[i, :], nchrom[i + 1, :])
        return nchrom.transpose()

    def mutate(self, chrom, pm):
        """
            变异
            :param chrom: 被选择个体
            :param pm: 变异概率
            :return: 变异后个体
            """
        nsel = len(chrom[0])
        nChrom = pd.DataFrame(chrom).transpose().to_numpy()
        l = len(chrom)
        for i in range(0, nsel):
            random = np.random.random()
            if pm > random:
                R = np.random.choice(l, l, replace=False)
                stepOne = R[0]
                stepTwo = R[1]
                t = nChrom[i][stepOne]
                nChrom[i][stepOne] = nChrom[i][stepTwo]
                nChrom[i][stepTwo] = t
        return nChrom.transpose()

    def relatedness(self, i, j, dist, vc):
        """
            求顾客i与顾客j之间的相关性
            :param i: 顾客
            :param j: 顾客
            :param dist: 距离矩阵
            :param vc: 每辆车所经过的顾客，用于判断i和j是否在一条路径上,如果在一条路径上为0，不在一条路径上为1
            :return: 顾客i和顾客j的相关性
            """
        # 顾客数量
        n = len(dist) - 1
        # 配送车辆数
        nv = len(vc)
        # 计算c[i][j]
        d = dist[i, j]
        maxDist = max(dist[i, 1:])
        maxIndex = np.where(dist[i] == maxDist)[0][0]
        c = d / maxDist
        # 判断i和j是否在一条路径上
        # 设初始顾客i与顾客j不在同一条路径上
        flag = 1
        for k in range(0, nv):
            # 该条路径上经过的顾客
            route = vc[k]
            # 判断该条路径上是否经过顾客i
            findi = np.where(route == j)[0]
            # 判断该条路径上是否经过顾客j
            findj = np.where(route == j)[0]
            # 如果findi和findj同时非空，则证明该条路径上同时经过顾客i和顾客j，则为0
            if len(findi) > 0 and len(findj) > 0:
                flag = 0
        return 1 / (c + flag)

    def remove(self, cusnum, toRemove, d, dist, vc):
        """
            Remove操作，先从原有顾客集合中随机选出一个顾客，然后根据相关性再依次移出需要数量的顾客
            :param cusnum: 顾客数量
            :param toRemove: 将要移出顾客的数量
            :param d: 随机元素
            :param dist: 距离矩阵
            :param vc: 每辆车所经过的顾客
            :return removed: 被移出的顾客集合
            :return rfvc: 移出顾客后的vc
            """
        removed = []
        # 所有顾客的集合
        inPlan = np.array(range(0, cusnum))
        # 随机从所有顾客中随机选出一个顾客
        visit = math.ceil(cusnum * (np.random.random()))
        inPlan = list(filter(lambda a: a != visit, inPlan))
        removed.append(visit)
        while len(removed) < toRemove:
            # 当前被移出的顾客数量
            nr = len(removed)
            # 从被移出的顾客集合中随机选择一个顾客
            rand = np.random.random()
            vr = math.ceil(nr * rand)
            # 原来顾客集合中顾客的数量
            nip = len(inPlan)
            # 存储removed(vr)与inplan中每个元素的相关性的数组
            R = np.zeros(nip)
            for i in range(0, nip):
                # 计算removed(vr)与inplan中每个元素的相关性
                R[i] = self.relatedness(int(removed[vr - 1]), int(inPlan[i]), dist, vc)
            # 降序
            SRV = np.sort(R)
            SRI = []
            i = 0
            while i < len(SRV):
                currIndex = np.where(R == SRV[i])[0]
                SRI = np.append(SRI, currIndex)
                if len(currIndex) > 1:
                    i = i + len(currIndex)
                else:
                    i = i + 1
            # 将inplan中的数组按removed(vr)与其的相关性从高到低排序
            lst = SRI
            # 从lst数组中选择一个客户
            rand = math.ceil(((np.random.random() ** d) * nip))
            nvc = lst[rand - 1]
            if len(np.where(removed == nvc)[0]) == 0:
                removed.append(int(nvc))
                inPlan = list(filter(lambda a: a != nvc, inPlan))
        # 移出removed中的顾客后的final_vehicles_customer
        rfvc = vc
        # # 最终被移出顾客的总数量
        nre = len(removed)
        # # 所用车辆数
        nv = len(vc)
        for i in range(0, nv):
            route = vc[i]
            for j in range(0, nre):
                # finrI = np.where(route == removed[j])[0]
                # if len(finrI):
                route = list(filter(lambda i: i != removed[j], route))
                rfvc[i] = np.array(route)

        def __init__(self):
            # 最大承重
            self.cap = 100
            # 违反的容量约束的惩罚函数系数
            self.alpha = 200
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

        # 更新染色体长度
        def update_choromsome_size(self, size):
            self.chromosome_size = size

        # 随机生成初始化种群
        def getinitialPopulation(self, length, populationSize):
            chromsomes = np.zeros((populationSize, length), dtype=np.int)
            newC = np.zeros((length, populationSize), dtype=np.int)
            for popusize in range(populationSize):
                # np.random.randit()产生[0,种群大小)之间的随机整数，第三个参数表示随机数的数量
                chromsomes[popusize, :] = np.random.randint(0, populationSize, length)
            for popusize in range(length):
                # np.random.randit()产生[0,种群大小)之间的随机整数，第三个参数表示随机数的数量
                newC[popusize, :] = np.random.choice(populationSize, populationSize, replace=False)
            return newC.transpose()

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

        def sus(self, fitness, nsel):
            """

                :param fitness: 个体的适应度值
                :param nsel: 被选择个体的数目
                :return: 被选择个体的索引号
                """
            nind = len(fitness)
            ans = np.ndim(fitness)
            # 执行随机常规采样
            cumfit = np.cumsum(fitness)
            # 平均适应度
            average = cumfit[-1] / nind
            # [0,1)随机数
            random = np.random.random()
            # nsel举证
            A = np.arange(start=0, stop=nsel, step=1)
            trials = average * (random + np.transpose(A))
            Mf = np.ones((nind, nsel))
            for i in range(0, nind):
                Mf[i] = cumfit[i]
            Mt = np.ones((nind, nsel))
            for i in range(0, nsel):
                Mt[::, i] = trials[i]
            Ma = np.zeros(nsel)
            Ma = np.insert(Mf[0:nind - 1, :], 0, Ma, axis=0)
            pMa = pd.DataFrame(Ma)
            pMf = pd.DataFrame(Mf)
            pMt = pd.DataFrame(Mt)
            pRf = pMt < pMf
            pRa = pMa <= pMt
            pRt = pd.DataFrame(pRf == pRa)
            NewChrIx = []
            ans = []
            shuf = []
            Rt = pRt.to_numpy()
            for i in range(0, len(Rt)):
                for j in range(0, len(Rt[i])):
                    if Rt[i][j] == True:
                        NewChrIx.append(i)
                        ans.append(Mt[i, j])
            randMt = np.random.rand(nsel, 1)
            srandMt = np.sort(randMt, axis=0)
            for i in range(0, len(srandMt)):
                shuf.append(np.where(randMt == srandMt[i])[0][0])
            # 根据shuf的序号取值
            resultIndex = []
            for i in range(0, len(shuf)):
                resultIndex.append(NewChrIx[shuf[i]])
            return resultIndex

        def select(self, chrom, fitness, ggap):
            """
                选择操作
                :param chrom:种群
                :param fitness:适应度值
                :param ggap:选择概率
                :return: 被选择的个体
                """
            # 种群数量
            nind = len(chrom[0])
            # 被选择个体的数目
            nsel = max(math.floor(nind * ggap + 0.5), 2)
            # 被选择个体的索引号
            chrIndex = self.sus(fitness, nsel)
            chr = pd.DataFrame(chrom).transpose()
            newChrom = []
            for i in chrIndex:
                item = chr.query('index ==' + str(i))
                newChrom.append(pd.DataFrame(item).to_numpy()[0])
            return pd.DataFrame(newChrom).transpose().to_numpy()

        def ox(self, a, b):
            """

                :param a: 待交换个体
                :param b: 待交换个体
                :return: 交叉后得到的两个个体
                """
            length = len(a)
            resultA = []
            resultB = []
            while 1:
                r1 = np.random.randint(length)
                r2 = np.random.randint(length)

                if r1 is not r2:
                    s = min(r1, r2)
                    e = max(r1, r2)
                    na = np.insert(a, 0, b[s:e + 1])
                    nb = np.insert(b, 0, a[s:e + 1], axis=0)
                    for i in range(0, len(na)):
                        aIndex = np.where(na == na[i])[0]
                        bIndex = np.where(nb == nb[i])[0]
                        if len(aIndex) > 1:
                            na[aIndex[1]] = -1
                        if len(bIndex) > 1:
                            nb[bIndex[1]] = -1
                        if (i + 1) == len(na):
                            break
                    for i in range(0, len(na)):
                        if na[i] != -1:
                            resultA.append(na[i])
                    for i in range(0, len(nb)):
                        if nb[i] != -1:
                            resultB.append(nb[i])
                    break
            return resultA, resultB

        def recombin(self, chrom, pc):
            """
                OX交换操作
                :param chrom: 被选择个体
                :param pc: 交叉概率
                :return: 交叉后的个体
                """
            # 被选择个体的数目
            nsel = len(chrom[0])
            nchrom = pd.DataFrame(chrom).transpose().to_numpy()
            end = nsel - np.mod(nsel, 2)
            for i in range(0, end, 2):
                # [0,1)随机数
                random = np.random.random()
                if pc >= random:
                    nchrom[i, :], nchrom[i + 1, :] = self.ox(nchrom[i, :], nchrom[i + 1, :])
            return nchrom.transpose()

        def mutate(self, chrom, pm):
            """
                变异
                :param chrom: 被选择个体
                :param pm: 变异概率
                :return: 变异后个体
                """
            nsel = len(chrom[0])
            nChrom = pd.DataFrame(chrom).transpose().to_numpy()
            l = len(chrom)
            for i in range(0, nsel):
                random = np.random.random()
                if pm > random:
                    R = np.random.choice(l, l, replace=False)
                    stepOne = R[0]
                    stepTwo = R[1]
                    t = nChrom[i][stepOne]
                    nChrom[i][stepOne] = nChrom[i][stepTwo]
                    nChrom[i][stepTwo] = t
            return nChrom.transpose()

        def relatedness(self, i, j, dist, vc):
            """
                求顾客i与顾客j之间的相关性
                :param i: 顾客
                :param j: 顾客
                :param dist: 距离矩阵
                :param vc: 每辆车所经过的顾客，用于判断i和j是否在一条路径上,如果在一条路径上为0，不在一条路径上为1
                :return: 顾客i和顾客j的相关性
                """
            # 顾客数量
            n = len(dist) - 1
            # 配送车辆数
            nv = len(vc)
            # 计算c[i][j]
            d = dist[i, j]
            maxDist = max(dist[i, 1:])
            maxIndex = np.where(dist[i] == maxDist)[0][0]
            c = d / maxDist
            # 判断i和j是否在一条路径上
            # 设初始顾客i与顾客j不在同一条路径上
            flag = 1
            for k in range(0, nv):
                # 该条路径上经过的顾客
                route = vc[k]
                # 判断该条路径上是否经过顾客i
                findi = np.where(route == j)[0]
                # 判断该条路径上是否经过顾客j
                findj = np.where(route == j)[0]
                # 如果findi和findj同时非空，则证明该条路径上同时经过顾客i和顾客j，则为0
                if len(findi) > 0 and len(findj) > 0:
                    flag = 0
            return 1 / (c + flag)

        def remove(self, cusnum, toRemove, d, dist, vc):
            """
                Remove操作，先从原有顾客集合中随机选出一个顾客，然后根据相关性再依次移出需要数量的顾客
                :param cusnum: 顾客数量
                :param toRemove: 将要移出顾客的数量
                :param d: 随机元素
                :param dist: 距离矩阵
                :param vc: 每辆车所经过的顾客
                :return removed: 被移出的顾客集合
                :return rfvc: 移出顾客后的vc
                """
            removed = []
            # 所有顾客的集合
            inPlan = np.array(range(0, cusnum))
            # 随机从所有顾客中随机选出一个顾客
            visit = math.ceil(cusnum * (np.random.random()))
            inPlan = list(filter(lambda a: a != visit, inPlan))
            removed.append(visit)
            while len(removed) < toRemove:
                # 当前被移出的顾客数量
                nr = len(removed)
                # 从被移出的顾客集合中随机选择一个顾客
                rand = np.random.random()
                vr = math.ceil(nr * rand)
                # 原来顾客集合中顾客的数量
                nip = len(inPlan)
                # 存储removed(vr)与inplan中每个元素的相关性的数组
                R = np.zeros(nip)
                for i in range(0, nip):
                    # 计算removed(vr)与inplan中每个元素的相关性
                    R[i] = self.relatedness(int(removed[vr - 1]), int(inPlan[i]), dist, vc)
                # 降序
                SRV = np.sort(R)
                SRI = []
                i = 0
                while i < len(SRV):
                    currIndex = np.where(R == SRV[i])[0]
                    SRI = np.append(SRI, currIndex)
                    if len(currIndex) > 1:
                        i = i + len(currIndex)
                    else:
                        i = i + 1
                # 将inplan中的数组按removed(vr)与其的相关性从高到低排序
                lst = SRI
                # 从lst数组中选择一个客户
                rand = math.ceil(((np.random.random() ** d) * nip))
                nvc = lst[rand - 1]
                if len(np.where(removed == nvc)[0]) == 0:
                    removed.append(int(nvc))
                    inPlan = list(filter(lambda a: a != nvc, inPlan))
            # 移出removed中的顾客后的final_vehicles_customer
            rfvc = vc
            # # 最终被移出顾客的总数量
            nre = len(removed)
            # # 所用车辆数
            nv = len(vc)
            for i in range(0, nv):
                route = vc[i]
                for j in range(0, nre):
                    # finrI = np.where(route == removed[j])[0]
                    # if len(finrI):
                    route = list(filter(lambda i: i != removed[j], route))
                    rfvc[i] = np.array(route)
            return removed, rfvc

        def cheapestIP(self, rv, rfvc, dist, demands, cap):
            """
                找出Removed数组中任一个元素的cheapest insertion point

                思路：
                第一步：先找出满足时间窗约束和容量约束的所有插入点，再计算上述插入点的距离增量
                第二步：找出上述插入点距离增量最小的那个最佳插入点，并记录距离增量

                :param rv: Removed数组中的任一个元素
                :param rfvc: 移出removed中的顾客后的final_vehicles_customer
                :param dist: 距离矩阵
                :param demands: 需求量
                :param cap: 最大载重量
                :return civ: 将rv插入到rfvc中在满足容量和时间窗约束下的距离增量最小的那辆车
                :return cip: 将rv插入到rfvc中在满足容量和时间窗约束下的距离增量最小的那辆车中的插入点
                :return C: 将rv插入到最佳插入点后的距离增量
                """
            # 所用车辆数量
            nv = len(rfvc)
            # 存储每一个合理的插入点以及对应的距离增量[车辆序号,插入点序号,距离增量]
            outcome = []
            for i in range(0, nv):
                # 其中一条路径
                route = rfvc[i]
                # 该路径上所经过顾客数量
                routeLen = len(route)
                # 插入rv之前该条路径的距离
                lb = self.part_length(route, dist)
                # 先将rv插入到route中的任何空隙，共(len + 1)个,
                for j in range(0, routeLen):
                    temp_r = None
                    # 将rv插入到集配中心后
                    if j == 0:
                        temp_r = np.insert(route, 0, rv)
                    # 将rv插入到集配中心前
                    elif j == (routeLen - 1):
                        temp_r = np.append(route, rv)
                    # 插入rv之后该条路径的距离
                    else:
                        temp_r = np.insert(route, j - 1, rv)
                    la = self.part_length(temp_r, dist)
                    # 插入rv之后该条路径的距离增量
                    delta = la - lb
                    # 判断一条路线是否满足载重量约束，1表示满足，0表示不满足
                    flag = self.judge_route(temp_r, demands, cap)
                    if flag == 1:
                        cur = [i, j, delta]
                        if outcome is None:
                            outcome = [cur]
                        else:
                            outcome = outcome.append(cur)
                # 如果存在合理的插入点，则找出最优插入点，否在新增加一辆车运输
                if outcome is not None and len(outcome) > 0:
                    # 每个插入点的距离增量
                    pdoutcome = pd.DataFrame(outcome)
                    addC = np.array(pdoutcome.loc[:, 2])
                    # 将距离增量从小到达排序
                    saC = np.sort(addC)
                    sIndex = []
                    for index in range(0, len(saC)):
                        sIndex.append(np.where(addC == saC[index])[0][0])
                    firstItem = sIndex[0]
                    return outcome[firstItem][0], outcome[firstItem][1], outcome[firstItem][2]
                else:
                    civ = nv + 1
                    cip = 1
                    C = self.part_length([rv], dist)
                    return civ, cip, C

        def farthestINS(self, removed, rfvc, dist, demands, cap):
            """
                最远插入启发式：将最小插入目标距离增量最大的元素找出来
                :param removed: 被移出的顾客集合
                :param rfvc:移出removed中的顾客后的final_vehicles_customer
                :param dist: 距离矩阵
                :param demands: 需求量
                :param cap: 最大载重量
                :return fv: 将removed中所有元素 最佳插入后距离增量最大的元素
                :return fviv: 该元素所插入的车辆
                :return fvip: 该元素所插入的车辆的坐标
                :return fvC: 该元素插入最佳位置后的距离增量
                """
            # 被移出的顾客的数量
            nr = len(removed)
            outcome = np.zeros((nr, 3))
            for i in range(0, nr):
                civ, cip, C = self.cheapestIP(removed[i], rfvc, dist, demands, cap)
                outcome[i][0] = civ
                outcome[i][1] = cip
                outcome[i][2] = C
            pdoutcome = pd.DataFrame(outcome)
            macIndex = pdoutcome.sort_values(by=2, ascending=False).index[0]
            mac = pdoutcome.loc[macIndex]
            fviv = int(pdoutcome.loc[macIndex, 0])
            fvip = int(pdoutcome.loc[macIndex, 1])
            fvC = pdoutcome.loc[macIndex, 2]
            fv = removed[macIndex]
            return fv, fviv, fvip, fvC

        def insert(self, fv, fviv, fvip, fvC, rfvc, dist):
            """
            根据插入点将元素插回到原始解中
            :param fv: 插回元素
            :param fviv: 将插回元素插回的车辆序号
            :param fvip: 将插回元素插回车辆序号中插入点的位置
            :param fvC:  该元素插入最佳位置后的距离增量
            :param rfvc: 移出removed中的顾客后的final_vehicles_customer
            :param dist: 距离矩阵
            :return ifvc: 插回元素后的rfvc
            :return iTD: 插回元素后的rfvc的总距离
            """
            ifvc = rfvc
            # 插回前的总距离
            sumTD = self.travel_distance(rfvc, dist)
            # 插回后的总距离
            iTD = sumTD + fvC
            # 如果插回车辆属于rfvc中的车辆
            if fviv < len(rfvc):
                route = rfvc[fviv]
                routeLen = len(route)
                if fvip == 0:
                    temp = np.insert(route, 0, fv)
                elif fvip == len:
                    temp = np.insert(route, -1, fv)
                else:
                    temp = np.insert(route, fvip - 1, fv)
                ifvc[fviv] = temp
            else:
                # 否则，新增加一辆车
                ifvc.append(np.array([fv]))
            return ifvc, iTD

        def re_insert(self, removed, rfvc, dist, demands, cap):
            """
                将被移出的顾客重新插回所得到的新的车辆顾客分配方案
                :param removed: 被移出的顾客集合
                :param rfvc: 移出removed中的顾客后的final_vehicles_customer
                :param dist: 距离矩阵
                :param demands: 需求量
                :param cap: 最大载重量
                :return ReIfvc: 将被移出的顾客重新插回所得到的新的车辆顾客分配方案
                :return RTD: 新分配方案的总距离
                """
            iRfvc = None
            while len(removed) > 0:
                # 最远插入启发式：将最小插入目标距离增量最大的元素找出来
                fv, fviv, fvip, fvC = self.farthestINS(removed, rfvc, dist, demands, cap);
                removed = list(filter(lambda a: a != fv, removed))
                # 根据插入点将元素插回到原始解中
                iRfvc, iTD = self.insert(fv, fviv, fvip, fvC, rfvc, dist)
            # 去除空路径
            ReIfvc = []
            for i in iRfvc:
                len(i) > 0 and ReIfvc.append(i)
            RTD = self.travel_distance(ReIfvc, dist)
            return ReIfvc, RTD

        def change(self, vc, n, cusnum):
            """
            这个函数有问题
            配送方案与个体之间进行转换
            :param vc: 每辆车经过的顾客
            :param n:
            :param cusnum:
            :return:
            """
            # 车辆使用数目
            nv = len(vc)
            chrom = []
            for i in range(0, nv):
                if (cusnum + i) < n:
                    if chrom == []:
                        chrom = vc[i]
                    else:
                        chrom = np.append(chrom, vc[i])
                    chrom = np.append(chrom, (cusnum + i))
                else:
                    if chrom == []:
                        chrom = vc[i]
                    else:
                        chrom = np.append(chrom, vc[i])
            # 如果染色体长度小于N，则需要向染色体添加配送中心编号
            if len(chrom) < n:
                supply = np.array(range((cusnum + nv), n))
                chrom = np.append(chrom, supply)
            return chrom

        def localSearch(self, chrom, cusnum, cap, demands, dist, alpha):
            """
                局部搜索函数
                :param chrom: 被选择的个体
                :param cusnum: 顾客数目
                :param cap: 最大载重量
                :param demands: 需求量
                :param dist: 距离矩阵 满足三角关系，暂用距离表示花费c[i][j]=dist[i][j]
                :param alpha: 进化逆转后的个体
                :return:
                """
            # Remove过程中的随机元素
            d = 15
            # 将要移出顾客的数量
            toRemove = min(math.ceil(cusnum / 2), 15)
            row = len(chrom[0])
            n = len(chrom)
            nChrom = pd.DataFrame(chrom).transpose().to_numpy()
            for i in range(0, row):
                vc, nv, td, violate_num, violate_cus = self.decode(chrom[:, i], cusnum, cap, demands, dist)
                cost = self.get_cost(vc, dist, demands, cap, alpha, td)
                removed, rfvc = self.remove(cusnum, toRemove, d, dist, vc)
                ReIfvc, RTD = self.re_insert(removed, rfvc, dist, demands, cap)
                # 计算惩罚函数
                RCF = self.get_cost(ReIfvc, dist, demands, cap, alpha, td)
                if RCF < cost:
                    nnChrom = self.change(ReIfvc, n, cusnum)

        return removed, rfvc

    def cheapestIP(self, rv, rfvc, dist, demands, cap):
        """
            找出Removed数组中任一个元素的cheapest insertion point

            思路：
            第一步：先找出满足时间窗约束和容量约束的所有插入点，再计算上述插入点的距离增量
            第二步：找出上述插入点距离增量最小的那个最佳插入点，并记录距离增量

            :param rv: Removed数组中的任一个元素
            :param rfvc: 移出removed中的顾客后的final_vehicles_customer
            :param dist: 距离矩阵
            :param demands: 需求量
            :param cap: 最大载重量
            :return civ: 将rv插入到rfvc中在满足容量和时间窗约束下的距离增量最小的那辆车
            :return cip: 将rv插入到rfvc中在满足容量和时间窗约束下的距离增量最小的那辆车中的插入点
            :return C: 将rv插入到最佳插入点后的距离增量
            """
        # 所用车辆数量
        nv = len(rfvc)
        # 存储每一个合理的插入点以及对应的距离增量[车辆序号,插入点序号,距离增量]
        outcome = []
        for i in range(0, nv):
            # 其中一条路径
            route = rfvc[i]
            # 该路径上所经过顾客数量
            routeLen = len(route)
            # 插入rv之前该条路径的距离
            lb = self.part_length(route, dist)
            # 先将rv插入到route中的任何空隙，共(len + 1)个,
            for j in range(0, routeLen):
                temp_r = None
                # 将rv插入到集配中心后
                if j == 0:
                    temp_r = np.insert(route, 0, rv)
                # 将rv插入到集配中心前
                elif j == (routeLen - 1):
                    temp_r = np.append(route, rv)
                # 插入rv之后该条路径的距离
                else:
                    temp_r = np.insert(route, j - 1, rv)
                la = self.part_length(temp_r, dist)
                # 插入rv之后该条路径的距离增量
                delta = la - lb
                # 判断一条路线是否满足载重量约束，1表示满足，0表示不满足
                flag = self.judge_route(temp_r, demands, cap)
                if flag == 1:
                    cur = [i, j, delta]
                    if outcome is None:
                        outcome = [cur]
                    else:
                        outcome = outcome.append(cur)
            # 如果存在合理的插入点，则找出最优插入点，否在新增加一辆车运输
            if outcome is not None and len(outcome) > 0:
                # 每个插入点的距离增量
                pdoutcome = pd.DataFrame(outcome)
                addC = np.array(pdoutcome.loc[:, 2])
                # 将距离增量从小到达排序
                saC = np.sort(addC)
                sIndex = []
                for index in range(0, len(saC)):
                    sIndex.append(np.where(addC == saC[index])[0][0])
                firstItem = sIndex[0]
                return outcome[firstItem][0], outcome[firstItem][1], outcome[firstItem][2]
            else:
                civ = nv + 1
                cip = 1
                C = self.part_length([rv], dist)
                return civ, cip, C

    def farthestINS(self, removed, rfvc, dist, demands, cap):
        """
            最远插入启发式：将最小插入目标距离增量最大的元素找出来
            :param removed: 被移出的顾客集合
            :param rfvc:移出removed中的顾客后的final_vehicles_customer
            :param dist: 距离矩阵
            :param demands: 需求量
            :param cap: 最大载重量
            :return fv: 将removed中所有元素 最佳插入后距离增量最大的元素
            :return fviv: 该元素所插入的车辆
            :return fvip: 该元素所插入的车辆的坐标
            :return fvC: 该元素插入最佳位置后的距离增量
            """
        # 被移出的顾客的数量
        nr = len(removed)
        outcome = np.zeros((nr, 3))
        for i in range(0, nr):
            civ, cip, C = self.cheapestIP(removed[i], rfvc, dist, demands, cap)
            outcome[i][0] = civ
            outcome[i][1] = cip
            outcome[i][2] = C
        pdoutcome = pd.DataFrame(outcome)
        macIndex = pdoutcome.sort_values(by=2, ascending=False).index[0]
        mac = pdoutcome.loc[macIndex]
        fviv = int(pdoutcome.loc[macIndex, 0])
        fvip = int(pdoutcome.loc[macIndex, 1])
        fvC = pdoutcome.loc[macIndex, 2]
        fv = removed[macIndex]
        return fv, fviv, fvip, fvC

    def insert(self, fv, fviv, fvip, fvC, rfvc, dist):
        """
        根据插入点将元素插回到原始解中
        :param fv: 插回元素
        :param fviv: 将插回元素插回的车辆序号
        :param fvip: 将插回元素插回车辆序号中插入点的位置
        :param fvC:  该元素插入最佳位置后的距离增量
        :param rfvc: 移出removed中的顾客后的final_vehicles_customer
        :param dist: 距离矩阵
        :return ifvc: 插回元素后的rfvc
        :return iTD: 插回元素后的rfvc的总距离
        """
        ifvc = rfvc
        # 插回前的总距离
        sumTD = self.travel_distance(rfvc, dist)
        # 插回后的总距离
        iTD = sumTD + fvC
        # 如果插回车辆属于rfvc中的车辆
        if fviv <= len(rfvc):
            route = rfvc[fviv]
            routeLen = len(route)
            if fvip == 0:
                temp = np.insert(route, 0, fv)
            elif fvip == len:
                temp = np.insert(route, -1, fv)
            else:
                temp = np.insert(route, fvip - 1, fv)
            ifvc[fviv] = temp
        else:
            # 否则，新增加一辆车
            ifvc.append(np.array([fv]))
        return ifvc, iTD

    def re_insert(self, removed, rfvc, dist, demands, cap):
        """
            将被移出的顾客重新插回所得到的新的车辆顾客分配方案
            :param removed: 被移出的顾客集合
            :param rfvc: 移出removed中的顾客后的final_vehicles_customer
            :param dist: 距离矩阵
            :param demands: 需求量
            :param cap: 最大载重量
            :return ReIfvc: 将被移出的顾客重新插回所得到的新的车辆顾客分配方案
            :return RTD: 新分配方案的总距离
            """
        iRfvc = None
        while len(removed) > 0:
            # 最远插入启发式：将最小插入目标距离增量最大的元素找出来
            fv, fviv, fvip, fvC = self.farthestINS(removed, rfvc, dist, demands, cap);
            removed = list(filter(lambda a: a != fv, removed))
            # 根据插入点将元素插回到原始解中
            iRfvc, iTD = self.insert(fv, fviv, fvip, fvC, rfvc, dist)
        # 去除空路径
        ReIfvc = []
        for i in iRfvc:
            len(i) > 0 and ReIfvc.append(i)
        RTD = self.travel_distance(ReIfvc, dist)
        return ReIfvc, RTD

    def change(self, vc, n, cusnum):
        """
        这个函数有问题
        配送方案与个体之间进行转换
        :param vc: 每辆车经过的顾客
        :param n:
        :param cusnum:
        :return:
        """
        # 车辆使用数目
        nv = len(vc)
        chrom = []
        for i in range(0, nv):
            if (cusnum + i + 1) < n:
                if len(chrom) == 0:
                    chrom = vc[i]
                else:
                    chrom = np.append(chrom, vc[i])
                chrom = np.append(chrom, int(cusnum + i + 1))
            else:
                if len(chrom) == 0:
                    chrom = vc[i]
                else:
                    chrom = np.append(chrom, vc[i])
        # 如果染色体长度小于N，则需要向染色体添加配送中心编号
        if len(chrom) < n:
            diff = []
            for i in range(0, n):
                if len(np.where(chrom == i)[0]) == 0:
                    diff.append(i)
            if len(diff) > 0:
                chrom = np.append(chrom, diff)
            else:
                supply = np.array(range((cusnum + nv), n))
                chrom = np.append(chrom, supply)
        return chrom

    def localSearch(self, chrom, cusnum, cap, demands, dist, alpha):
        """
            局部搜索函数
            :param chrom: 被选择的个体
            :param cusnum: 顾客数目
            :param cap: 最大载重量
            :param demands: 需求量
            :param dist: 距离矩阵 满足三角关系，暂用距离表示花费c[i][j]=dist[i][j]
            :param alpha: 进化逆转后的个体
            :return:
            """
        # Remove过程中的随机元素
        d = 15
        # 将要移出顾客的数量
        toRemove = min(math.ceil(cusnum / 2), 15)
        row = len(chrom[0])
        n = len(chrom)
        nChrom = pd.DataFrame(chrom).transpose().to_numpy()
        for i in range(0, row):
            vc, nv, td, violate_num, violate_cus = self.decode(nChrom[i, :], cusnum, cap, demands, dist)
            cost = self.get_cost(vc, dist, demands, cap, alpha, td)
            removed, rfvc = self.remove(cusnum, toRemove, d, dist, vc)
            ReIfvc, RTD = self.re_insert(removed, rfvc, dist, demands, cap)
            # 计算惩罚函数
            RCF = self.get_cost(ReIfvc, dist, demands, cap, alpha, td)
            if RCF < cost:
                nnChrom = self.change(ReIfvc, n, cusnum)
                nChrom[i, :] = nnChrom
        return nChrom.transpose()

    def reins(self, chromosome, selectCh, cost):
        """
        重插入子代的新种群
        :param chromosome: 父代的种群
        :param selectCh: 子代种群
        :param cost: 父代适应度
        :return: 组合父代与子代后得到的新种群
        """
        nind = len(chromosome[0])
        nsel = len(selectCh[0])
        tChromsome = pd.DataFrame(chromosome).transpose().to_numpy()
        tSelectch = pd.DataFrame(selectCh).transpose().to_numpy()
        sortCost = np.sort(cost)
        index = []
        for i in sortCost:
            curIndex = np.where(i == cost)[0][0]
            index.append(curIndex)
        final = nind - nsel
        if final == nind:
            final = -1
        newIndex = index[0:nind - nsel]
        beforChromsome = []
        for i in newIndex:
            beforChromsome.append(tChromsome[i])
        for i in tSelectch:
            beforChromsome.append(i)
        return pd.DataFrame(beforChromsome).transpose().to_numpy()

    def del_repeat(self, chromosome):
        """
        删除种群中重复个体，并补齐删除的个体
        :param chromosome: 整理前个体
        :return: 整理后个体
        """
        tChromosome = pd.DataFrame(chromosome).transpose()
        nind = len(chromosome[0])
        length = len(chromosome)
        dChrom = tChromosome.drop_duplicates(ignore_index=True)
        nd = len(dChrom.to_numpy())
        nChrom = None
        if nind > nd:
            nChrom = self.getinitialPopulation(nind - nd, length)
        if nChrom is None:
            return dChrom.transpose().to_numpy()
        else:
            tnChrom = np.array(nChrom).transpose()
            return np.insert(dChrom.to_numpy(), len(dChrom.to_numpy()), tnChrom, axis=0).transpose()
