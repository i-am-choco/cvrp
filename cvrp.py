"""
@author：choco
@description：gacvrp处理主类
"""
from traceback import print_tb

from importData import ImportData
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


class Cvrp:
    """ 车载路径处理主模块 """

    def __init__(self):
        """构造函数，初始化属性"""
        # 导入数据
        self.importData = ImportData()
        original = self.importData.get_original_data(None)
        # 所有坐标数据
        self.points = self.importData.get_coordinate_point(original)
        # 坐标信息
        self.customerRes = self.importData.get_customer_points(self.points)
        # 需求数
        self.demands = self.importData.get_demands(original)
        # 顾客坐标
        self.customer = self.customerRes.get('customerList')
        # 顾客总数
        self.customerLen = self.customerRes.get('customerLen')
        # 配送中心坐标
        self.distribution = self.customerRes.get('distribution')
        # 车辆使用数量
        self.vehicle = 25

    # 计算各行间向量欧式距离
    # D=[d12,d13,....,d1n,d23,d24,...d2n,....,d(n-1)dn],
    # len = m(m-1)/2 ,其中m是坐标总数，即行数
    def get_distance(self, data):
        if data is not None:
            return pdist(np.arry(data))
        return pdist(np.array(self.points))

    # 获取坐标换算的距离矩阵
    def get_distance_matrix(self, data):
        if data is not None:
            return squareform(data)
        return squareform(self.get_distance(None))

    # 绘制成本图
    def get_plot_best_cost(self, cost):
        plt.figure(1)
        plt.plot(cost)
        plt.title('Plot 1')

    # 繪製坐標圖
    def get_plot_coordinates(self):
        x_coords = [point[0] for point in self.customer]
        y_coords = [point[1] for point in self.customer]
        plt.figure(2)
        plt.scatter(self.distribution[0], self.distribution[1], color='red', s=100, zorder=5, label="Red Point")
        plt.scatter(x_coords, y_coords, color='b', label='Points', marker='o')
        plt.title('XY Coordinates Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.grid()


    def get_plot_route_map(self,routes):
        plt.figure(3)
        colors = plt.cm.get_cmap('tab20', len(routes))
        for i, route in enumerate(routes):
            route_coords = self.customer[route-1]  # 获取路线的坐标
            route_coords = np.vstack([self.distribution, route_coords, self.distribution])
            x, y = route_coords[:, 0], route_coords[:, 1]  # 拆分为x和y
            plt.plot(x, y, marker='o', label=f'Route {i + 1}', color=colors(i))
            plt.text(x[0], y[0], "", color='green', fontsize=9)  # 起点标记
            plt.text(x[-1], y[-1], "", color='red', fontsize=9)  # 终点标记
        plt.legend()
        plt.title("VRP Route Map")
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.grid()

