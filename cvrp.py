"""
@author：choco
@description：gacvrp处理主类
"""

from importData import ImportData
import numpy as np
from scipy.spatial.distance import pdist, squareform


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


