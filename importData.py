"""
@author：choco
"""
import numpy as np
import pandas as pd


class ImportData:
    """
        获取数据函数
        默认文本文件数据解说
        每列含义： 【序号 X坐标 Y坐标 需求量】
        其中第一行数据表示配送中心地点 所以第一行数据需求量为0；
    """

    # 获取原始文件数据
    @staticmethod
    def get_original_data(path=None):
        filePath = None
        if path is not None:
            filePath = path
        # 判断是否传入导入数据文件地址，若无则使用默认demo文件
        else:
            filePath = 'dataSet/rc208.xlsx'
        print('currentPath:' + filePath)
        data = pd.read_excel(filePath, sheet_name='Sheet1')
        return data

    # 提取所有坐标点
    def get_coordinate_point(self, data):
        if data is not None:
            return data[['x', 'y']].values
        else:
            original_data = self.get_original_data()
            return self.get_coordinate_point(original_data)

    def get_demands(self, data):
        if data is not None:
            return data['num'].values
        else:
            original_data = self.get_original_data()
            return self.get_coordinate_point(original_data)


    # 提取顾客坐标信息等
    def get_customer_points(self, data: np.ndarray):
        res = {
            "distribution": None,
            "customerList": [],
            "customerLen": 0,
        };
        if data is not None:
            res['distribution'] = data[0]
            res['customerList'] = data[1:]
            res['customerLen'] = len(data) - 1
            return res
        else:
            points_data = self.get_coordinate_point(None)
            return self.get_customer_points(points_data)

