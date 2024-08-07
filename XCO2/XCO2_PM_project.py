# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:27:32 2023

@author: 21
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 12:43:10 2023

@author: : wlx
提取波段数据然后计算相应的指数

"""

import re
from pyproj import Proj, transform
from osgeo import gdal, osr
import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xml.etree.ElementTree import ElementTree, Element
import warnings
from skimage import transform as TF_resize
warnings.filterwarnings('ignore')


# 单波段xls经纬度点位提取main Function

# 增加换行符
def __indent(elem, level=0):
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


class ArgParser:
    def __init__(self, xml_path):
        self.xml_path = xml_path

    def parse(self):
        xml_tree = ElementTree()
        xml_tree.parse(self.xml_path)
        xml_root = xml_tree.getroot()
        param_dict = {}
        for param in xml_root:
            param_name = param.attrib['name']
            value = param[0]
            param_value = value.text
            try:
                param_dict[param_name] = eval(param_value)
            except:
                # 当param_value无法转换为其它类型时 采用初始值
                param_dict[param_name] = param_value
        return param_dict


# 栅格数据读写

class IMAGE:
    # 读图像文件
    def read_img(self, filename):
        dataset = gdal.Open(filename)  # 打开文件
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_bands = dataset.RasterCount  # 波段数
        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
        im_proj = dataset.GetProjection()  # 地图投影信息，字符串表示
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

        del dataset

        return im_width, im_height, im_bands, im_proj, im_geotrans, im_data

    # 写GeoTiff文件
    def write_img(self, filename, im_proj, im_geotrans, im_data):

        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

        del dataset


# 1）基于经纬度坐标提取影像信息：关键函数
def Bands_XlsLatLon_Extraction(xls_path, tif_path):
    # 读取Excel格式文件,.xls和.xlsx两种格式均可

    # 经纬度坐标信息读取
    data = pd.read_excel(xls_path)
    lon = list(data.iloc[:, 0])
    lat = list(data.iloc[:, 1])

    # 坐标转为图上行列号
    # 判断当前影像是投影坐标,还是地理坐标，如果是投影坐标,需要将点位(经纬度)进行转换
    dataset = gdal.Open(tif_path)
    tif_geo = dataset.GetGeoTransform()  # 获取六参数的(左上角横坐标,水平空间分辨率,旋转值一般为0,左上角纵坐标,旋转值一般为0,垂直分辨率)元组
    data = dataset.ReadAsArray().astype(np.float64)  # 获取数据
    im_bands = dataset.RasterCount  # 波段数
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    print("波段数：", im_bands)
    if len(data.shape) == 2:
        data = np.reshape(data, (1, data.shape[0], data.shape[1]))

    number = len(lon)  # 循环次数
    row = []
    column = []
    valid_count = []
    prj = dataset.GetProjection()
    osrobj = osr.SpatialReference()
    osrobj.ImportFromWkt(prj)
    if osrobj.IsProjected():
        print("***********影像是投影坐标系，需要对输入经纬度进行转换***********")
        # 经纬度的点进行投影转换
        lon_to_mapx, lat_to_mapy = Prj_PointValue(lon, lat, prj)
        # utm = Proj(proj='utm', zone=47, ellps='WGS84')
        for i in range(number):
            # map_x, map_y = utm(lon[i],lat[i])
            map_x = lon_to_mapx[i]
            map_y = lat_to_mapy[i]
            # 获取点位坐标, 行和列
            point_row, point_column = index_PointValue(map_x, map_y, tif_geo)
            # 因为转换行列号后超出了范围，所以出现了一些问题，但是在筛选过程中已经选在范围内的了
            valid_point = 0 <= point_row < im_height and 0 <= point_column < im_width
            if valid_point:
                row.append(point_row)
                column.append(point_column)
            else:
                valid_count.append(i)
                row.append(-1)
                column.append(-1)

    elif osrobj.IsGeographic():
        print("***********影像是地理坐标系***********")
        for i in range(number):
            # 获取点位坐标, 行和列；有些坐标会超出范围，所以要剔除：valid_count
            point_row, point_column = index_PointValue(lon[i], lat[i], tif_geo)
            valid_point = 0 <= point_row < im_height and 0 <= point_column < im_width
            if valid_point:
                row.append(point_row)
                column.append(point_column)
            else:
                valid_count.append(i)
                row.append(-1)
                column.append(-1)


    else:
        print("坐标系判断出现问题！！！")
    # 根据行列号提取影像数值
    PointValue = data[:, row, column].T
    PointValue[valid_count, :] = -1

    return PointValue, valid_count

    # 当目标影像是投影坐标系时,对经纬度的点进行投影转换


def Prj_PointValue(lon, lat, prj):
    # DeprecationWarning: This function is deprecated.显示警告信息
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # 利用目标影像截取投影信息
    str_list = re.compile(r'"(.*?)"').findall(prj)  # 利用正则表达式选取冒号里面的内容
    coordinate = str_list[-2] + ":" + str_list[-1]

    point_coordinate = Proj(init="epsg:4326")  # 定义数据地理坐标系
    target_coordinate = Proj(init=coordinate)  # 定义与目标影像相同的投影坐标系
    x1, y1 = point_coordinate(lon, lat)
    lon_to_mapx, lat_to_mapy = transform(point_coordinate, target_coordinate, x1, y1)
    return lon_to_mapx, lat_to_mapy

    # 获取点位所在的行和列（索引）


def index_PointValue(X, Y, tif_geo):
    point_row = int((tif_geo[3] - Y) / tif_geo[1])
    point_column = int((X - tif_geo[0]) / tif_geo[1])
    return point_row, point_column


# 2）简单的提取之后构建影响因素指标：关键函数
def sentinel2_pm_bandmath_xls(banddata_xls, idx):
    # banddata_xls：[lon,lat,pm,b1-b12,tem,wind]
    # 根据excle文件计算相应的指数；xls数据读取索引从0开始
    background_value = 0
    # p
    figure = idx + 1
    #
    start_idx = 3
    # banddata_xls = pd.read_excel( xls_path )

    if figure <= 12:
        out_data = banddata_xls[:, idx + 3] * 0.0001
    elif figure == 13:
        b4 = banddata_xls[:, start_idx + 3]
        b8 = banddata_xls[:, start_idx + 7]
        out_data = 1.0 * (b8 - b4) / (b8 + b4)
        nan_idx = np.isnan(out_data)
        out_data[nan_idx] = background_value
        # out_data = out_data.astype(np.float32)
    elif figure == 14:
        b5 = banddata_xls[:, start_idx + 4]
        b8 = banddata_xls[:, start_idx + 7]
        out_data = 1.0 * (b8 - b5) / (b8 + b5)
        nan_idx = np.isnan(out_data)
        out_data[nan_idx] = background_value
    elif figure == 15:
        b6 = banddata_xls[:, start_idx + 5]
        b8 = banddata_xls[:, start_idx + 7]
        out_data = 1.0 * (b8 - b6) / (b8 + b6)
        nan_idx = np.isnan(out_data)
        out_data[nan_idx] = background_value
    elif figure == 16:
        b7 = banddata_xls[:, start_idx + 6]
        b8 = banddata_xls[:, start_idx + 7]
        out_data = 1.0 * (b8 - b7) / (b8 + b7)
        nan_idx = np.isnan(out_data)
        out_data[nan_idx] = background_value
    elif figure == 17:
        b4 = banddata_xls[:, start_idx + 3]
        b9 = banddata_xls[:, start_idx + 8]
        out_data = 1.0 * (b9 - b4) / (b9 + b4)
        nan_idx = np.isnan(out_data)
        out_data[nan_idx] = background_value
    elif figure == 18:
        b5 = banddata_xls[:, start_idx + 4]
        b9 = banddata_xls[:, start_idx + 8]
        out_data = 1.0 * (b9 - b5) / (b9 + b5)
        nan_idx = np.isnan(out_data)
        out_data[nan_idx] = background_value
    elif figure == 19:
        b6 = banddata_xls[:, start_idx + 5]
        b9 = banddata_xls[:, start_idx + 8]
        out_data = 1.0 * (b9 - b6) / (b9 + b6)
        nan_idx = np.isnan(out_data)
        out_data[nan_idx] = background_value
    elif figure == 20:
        b7 = banddata_xls[:, start_idx + 6]
        b9 = banddata_xls[:, start_idx + 8]
        out_data = 1.0 * (b9 - b7) / (b9 + b7)
        nan_idx = np.isnan(out_data)
        out_data[nan_idx] = background_value
    # tem-b13
    elif figure == 21:
        out_data = banddata_xls[:, start_idx + 12]
    # wind-b14
    elif figure == 22:
        out_data = banddata_xls[:, start_idx + 13]
    else:
        print('无效指数！')
    return out_data


class BandData_Extract_math:


    # 属性

    def __init__(self, p_point_xls_path: str, p_Sentinel2_path: str, p_tem_path: str, p_wind_path: str, p_out_xls: str):
        self._point_xls_path = p_point_xls_path
        self._Sentinel2_path = p_Sentinel2_path
        self._tem_path = p_tem_path
        self._wind_path = p_wind_path
        self._out_xls = p_out_xls

    def exit_error(self):
        """程序错误退出"""
        sys.exit(3)

    def exit_success(self):
        """程序成功退出"""
        sys.exit(0)

    def check_params(self):
        """
        参数检查，验证各个参数是否符合算法要求
            函数内部逻辑：
                1. 验证输入目录是否存在
                2. 不存在则创建输出目录，并验证是否创建成功
                3. 对参数"param1"的类型进行强制转换，保证获取的参数类型正确，并捕获对应异常
        """
        if not os.path.exists(self._Sentinel2_path):
            print('输入文件不存在:{0}'.format(self._Sentinel2_path))
            self.exit_error()

    def model_build(self):
        """
        程序核心处理逻辑
        处理逻辑
            依次输出三个参数值
            行68，为模拟处理过程中的异常，可通过取消注释触发异常
        """
        self.check_params()

        point_xls_path = self._point_xls_path
        Sentinel2_path = self._Sentinel2_path
        tem_path = self._tem_path
        wind_path = self._wind_path
        out_xls = self._out_xls

        paths = [Sentinel2_path, tem_path, wind_path]
        pm_data = pd.read_excel(point_xls_path).iloc[:, 0:3]
        extract_data = np.array(pm_data)
        for path in paths:
            #print(path)
            point_value, valid_idx = Bands_XlsLatLon_Extraction(point_xls_path, path)
            extract_data = np.append(extract_data, point_value, axis=1)

            # 规定输入是从1开始，因为b1-b22,这一步还是需要输出相关系数排名的。
        title_list = ["经度", "纬度", "pm值"]
        idx_num = 22

        # 数据读取后为什么要转换啊
        for i in range(idx_num):
            title_list.append("band" + str(i + 1) + "_value")
            add_data = sentinel2_pm_bandmath_xls(extract_data, i)
            # 需要转换一下维数
            add_data = add_data[:, np.newaxis]
            pm_data = np.append(pm_data, add_data, axis=1)
        pm_data[valid_idx, 3:] = -1
        df = pd.DataFrame(pm_data, columns=title_list)
        df.to_excel(out_xls, index=None, encoding="gbk")
        # assert 1 == 0, '模拟算法处理过程中的异常'



def PM_LinearRegression(xls_path, inmodel_idx, train_size):
    pm_data = pd.read_excel(xls_path)
    inmodel_idx = np.array(inmodel_idx)
    pm_data = np.array(pm_data)
    size = pm_data.shape
    end_idx = int(size[0] * train_size)

    # 选择了1，2，（b1 b2）,实际的列数（0开始）是3，4所以有这么个转换关系
    x_train = pm_data[0:(end_idx - 1), inmodel_idx + 2]
    y_train = pm_data[0:(end_idx - 1), 2]

    model = LinearRegression()
    model.fit(x_train, y_train)
    a = model.intercept_  # 截距
    b = model.coef_  # 回归系数
    print('The model is complete!')
    #print(model.predict(x_train))
    return a, b


class LinearModel_build:
    # 算法参数定义
    ParamName_Model = 'Model'
    ParamName_input_xls_path = 'input_xls_path'
    ParamName_inmodel_idx = 'inmodel_idx'
    ParamName_train_size = 'train_size'
    ParamName_valid_out_xls = 'valid_out_xls'
    ParamName_model_record_xls = 'model_record_xls'

    # 属性

    def __init__(self, p_Model: str, p_input_xls_path: str, p_inmodel_idx: str, p_train_size: str, p_valid_out_xls: str,
                 p_model_record_xls: str):

        self._Model = p_Model
        self._input_xls_path = p_input_xls_path
        self._inmodel_idx = p_inmodel_idx
        self._train_size = p_train_size
        self._valid_out_xls = p_valid_out_xls
        self._model_record_xls = p_model_record_xls

    def exit_error(self):
        """程序错误退出"""
        sys.exit(3)

    def exit_success(self):
        """程序成功退出"""
        sys.exit(0)

    def check_params(self):
        """
        参数检查，验证各个参数是否符合算法要求
            函数内部逻辑：
                1. 验证输入目录是否存在
                2. 验证输出目录是否存在
                    1. 不存在则创建输出目录，并验证是否创建成功
                3. 对参数"param1"的类型进行强制转换，保证获取的参数类型正确，并捕获对应异常
        """
        if not os.path.exists(self._input_xls_path):
            print('输入文件不存在:{0}'.format(self._input_xls_path))
            self.exit_error()

    def model_build(self):
        """
        程序核心处理逻辑
        处理逻辑
            依次输出三个参数值
            行68，为模拟处理过程中的异常，可通过取消注释触发异常
        """

        xls_path = self._input_xls_path
        inmodel_idx = self._inmodel_idx
        train_size = self._train_size
        valid_xls = self._valid_out_xls
        model_record_xls = self._model_record_xls

        # a = model2.intercept_   # 截距
        # b = model2.coef_        # 回归系数
        a, b = PM_LinearRegression(xls_path, inmodel_idx, train_size)

        #print("最佳拟合曲线：截距", a, ",回归系数：", b)

        pm_data = pd.read_excel(xls_path)
        sdata = np.array(pm_data)

        # 1,2->b1-b2（列的索引3，4）
        inmodel_idx = np.array(inmodel_idx)
        x_valid = sdata[:, inmodel_idx + 2]
        y_valid = np.dot(x_valid, b) + a
        y_real = sdata[:, 2]
        percent = 1 - abs((y_valid - y_real) / y_real)
        y_real = y_real[:, np.newaxis]
        y_valid = y_valid[:, np.newaxis]
        percent = percent[:, np.newaxis]

        vdata = np.concatenate((y_real, y_valid, percent), axis=1)
        vf = pd.DataFrame(vdata)
        vf.to_excel(valid_xls, index=None, encoding="gbk")

        # 模型记录输出文档
        inmodel_idx = inmodel_idx[:, np.newaxis]
        b = b[:, np.newaxis]
        sa = np.zeros((len(b), 1))
        sa[0] = a
        model_data = np.concatenate((inmodel_idx, b, sa), axis=1)
        title_list = ['band', 'ratios', 'delta']
        model_data = pd.DataFrame(model_data, columns=title_list)
        model_data.to_excel(model_record_xls, index=None, encoding="gbk")

        # assert 1 == 0, '模拟算法处理过程中的异常'


def Sentinel2_PM_Index(s2_path, tem_path, wind_path, idx):
    background_value = 0
    # gdal的波段索引从1开始
    figure = int(idx)

    # out_data =  raster_dataset.GetRasterBand(1).ReadAsArray()* scale
    # TypeError: in method 'Dataset_GetRasterBand', argument 2 of type 'int'
    # 因为这个错误只能把下面的idx给改了，Int32怎么都改不了
    raster_dataset = gdal.Open(s2_path)
    # nl = 10296
    # ns = 8517

    test_data = raster_dataset.GetRasterBand(1).ReadAsArray()
    height, width = test_data.shape
    scale = 0.0001
    if figure <= 12:
        out_data = raster_dataset.GetRasterBand(figure).ReadAsArray() * scale
    elif figure == 13:
        b4 = raster_dataset.GetRasterBand(4).ReadAsArray() * scale
        b8 = raster_dataset.GetRasterBand(8).ReadAsArray() * scale
        out_data = 1.0 * (b8 - b4) / (b8 + b4)
        nan_idx = np.isnan(out_data)
        out_data[nan_idx] = background_value
        # out_data = out_data.astype(np.float32)
    elif figure == 14:
        b5 = raster_dataset.GetRasterBand(5).ReadAsArray() * scale
        b8 = raster_dataset.GetRasterBand(8).ReadAsArray() * scale
        out_data = 1.0 * (b8 - b5) / (b8 + b5)
        nan_idx = np.isnan(out_data)
        out_data[nan_idx] = background_value
    elif figure == 15:
        b6 = raster_dataset.GetRasterBand(6).ReadAsArray() * scale
        b8 = raster_dataset.GetRasterBand(8).ReadAsArray() * scale
        out_data = 1.0 * (b8 - b6) / (b8 + b6)
        nan_idx = np.isnan(out_data)
        out_data[nan_idx] = background_value
    elif figure == 16:
        b7 = raster_dataset.GetRasterBand(7).ReadAsArray() * scale
        b8 = raster_dataset.GetRasterBand(8).ReadAsArray() * scale
        out_data = 1.0 * (b8 - b7) / (b8 + b7)
        nan_idx = np.isnan(out_data)
        out_data[nan_idx] = background_value
    elif figure == 17:
        b4 = raster_dataset.GetRasterBand(4).ReadAsArray() * scale
        b9 = raster_dataset.GetRasterBand(9).ReadAsArray() * scale
        out_data = 1.0 * (b9 - b4) / (b9 + b4)
        nan_idx = np.isnan(out_data)
        out_data[nan_idx] = background_value
    elif figure == 18:
        b5 = raster_dataset.GetRasterBand(5).ReadAsArray() * scale
        b9 = raster_dataset.GetRasterBand(9).ReadAsArray() * scale
        out_data = 1.0 * (b9 - b5) / (b9 + b5)
        nan_idx = np.isnan(out_data)
        out_data[nan_idx] = background_value
    elif figure == 19:
        b6 = raster_dataset.GetRasterBand(6).ReadAsArray() * scale
        b9 = raster_dataset.GetRasterBand(9).ReadAsArray() * scale
        out_data = 1.0 * (b9 - b6) / (b9 + b6)
        nan_idx = np.isnan(out_data)
        out_data[nan_idx] = background_value
    elif figure == 20:
        b7 = raster_dataset.GetRasterBand(7).ReadAsArray() * scale
        b9 = raster_dataset.GetRasterBand(9).ReadAsArray() * scale
        out_data = 1.0 * (b9 - b7) / (b9 + b7)
        nan_idx = np.isnan(out_data)
        out_data[nan_idx] = background_value
    # tem
    elif figure == 21:
        tem_raster_dataset = gdal.Open(tem_path)
        data = tem_raster_dataset.GetRasterBand(1).ReadAsArray()
        # wind（进行重采样的一个函数，保证行列号一致）
        out_data  = TF_resize.resize(data, (height, width), order=1)
    elif figure == 22:
        wind_raster_dataset = gdal.Open(wind_path)
        data = wind_raster_dataset.GetRasterBand(1).ReadAsArray()
        out_data = TF_resize.resize(data, (height, width), order=1)
    else:
        print('无效指数！')
    return out_data


class PMLinearModel_apply:

    # 属性
    def __init__(self, p_model_record_xls: str, p_Sentinel2_path: str, p_temp_path: str, p_wind_path: str,
                 p_output_tif: str):

        self._model_record_xls = p_model_record_xls
        self._Sentinel2_path = p_Sentinel2_path
        self._temp_path = p_temp_path
        self._wind_path = p_wind_path
        self._output_tif = p_output_tif

    def exit_error(self):
        """程序错误退出"""
        sys.exit(3)

    def exit_success(self):
        """程序成功退出"""
        sys.exit(0)

    def check_params(self):
        """
        参数检查，验证各个参数是否符合算法要求
            函数内部逻辑：
                1. 验证输入目录是否存在
                2. 验证输出目录是否存在
                    1. 不存在则创建输出目录，并验证是否创建成功
                3. 对参数"param1"的类型进行强制转换，保证获取的参数类型正确，并捕获对应异常
        """
        if not os.path.exists(self._model_record_xls):
            print('输入文件不存在:{0}'.format(self._model_record_xls))
            self.exit_error()

    def model_build(self):
        """
        程序核心处理逻辑
        处理逻辑
            依次输出三个参数值
            行68，为模拟处理过程中的异常，可通过取消注释触发异常
        """

        model_record_xls = self._model_record_xls
        s2_path = self._Sentinel2_path
        tem_path = self._temp_path
        wind_path = self._wind_path
        output_tif = self._output_tif

        df = pd.read_excel(model_record_xls)
        model_data = np.array(df)
        inmodel_idx = model_data[:, 0]
        a = model_data[0, 2]  # 截距
        b = model_data[:, 1]  # 系数

        # 将模型应用至空间数据:将三维数据转为二维，利用model.fit实现模型应用，最后对结果reshape
        tiff = IMAGE()
        # read_img函数返回值：im_width, im_height, im_bands, im_proj, im_geotrans, im_data
        tif = tiff.read_img(s2_path)
        # tif tuple 元组类型，一旦初始化后就不能修改
        shp = tif[5].shape
        # shp =(12,4793,3857) nb,nl,ns
        bandmath_data = np.zeros((shp[1], shp[2], len(inmodel_idx)))
        for j in range(len(inmodel_idx)):
            # 为什么in method 'Dataset_GetRasterBand', argument 2 of type 'int'
            # inmodel_idx[j]就老是出错呢
            bandmath_data[:, :, j] = Sentinel2_PM_Index(s2_path, tem_path, wind_path, inmodel_idx[j])
        # 对影像非有效值的区域赋值为零，或者NoData
        idx = bandmath_data[:, :, 0] <= 0
        bandmath_data = bandmath_data.reshape(shp[1] * shp[2], len(inmodel_idx))
        b = b[:, np.newaxis]
        data_predict = np.dot(bandmath_data, b) + a

        result = np.array(data_predict).reshape(shp[1], shp[2])
        result[idx] = np.nan
        # write_img(self, filename, im_proj, im_geotrans, im_data)
        tiff.write_img(output_tif, tif[3], tif[4], result)


class CO2_Model_apply:

    # 属性
    def __init__(self, p_pm_path: str, p_tem_path: str, p_rainfall_path: str, p_output_tif: str):

        self._pm_path = p_pm_path
        self._tem_path = p_tem_path
        self._rainfall_path = p_rainfall_path
        self._output_tif = p_output_tif

    def exit_error(self):
        """程序错误退出"""
        sys.exit(3)

    def exit_success(self):
        """程序成功退出"""
        sys.exit(0)

    def check_params(self):
        """
        参数检查，验证各个参数是否符合算法要求
            函数内部逻辑：
                1. 验证输入目录是否存在
                2. 验证输出目录是否存在
                    1. 不存在则创建输出目录，并验证是否创建成功
                3. 对参数"param1"的类型进行强制转换，保证获取的参数类型正确，并捕获对应异常
        """
        if not os.path.exists(self._pm_path):
            print('输入文件不存在:{0}'.format(self._pm_path))
            self.exit_error()

    def model_build(self):
        """
        程序核心处理逻辑
        处理逻辑
            依次输出三个参数值
            行68，为模拟处理过程中的异常，可通过取消注释触发异常
        """

        pm_path = self._pm_path
        rainfall_path = self._rainfall_path
        tem_path = self._tem_path
        output_tif = self._output_tif

        # 线性模型
        a = 413.836  # 截距
        b = np.array([0.0438088, -0.153728, 0])  # 系数

        # 数据读取：经过处理后的数据维度、范围保持一致
        tiff = IMAGE()
        # read_img函数返回值：im_width, im_height, im_bands, im_proj, im_geotrans, im_data
        im_width, im_height, im_bands, im_proj, im_geotrans, pm_data = tiff.read_img(pm_path)
        _, _, _, _, _, tem_data = tiff.read_img(tem_path)
        _, _, _, _, _, rainfall_data = tiff.read_img(rainfall_path)

        # 数据处理：维度展开
        # bandmath_data = np.zeros((im_width, im_height, len(b)))
        bandmath_data = np.zeros((im_height, im_width, len(b)))
        bandmath_data[:, :, 0] = pm_data
        bandmath_data[:, :, 1] = TF_resize.resize(tem_data, (im_height, im_width), order=1)
        #ERA5需要处理：bandmath_data[:, :, 1] = tem_data - 260
        bandmath_data[:, :, 2] = TF_resize.resize(rainfall_data, (im_height, im_width), order=1)
        idx = pm_data <= 0
        bandmath_data = bandmath_data.reshape(im_width * im_height, len(b))
        b = b[:, np.newaxis]
        data_predict = np.dot(bandmath_data, b) + a

        result = np.array(data_predict).reshape(im_height, im_width)
        result[idx] = np.nan
        result[result > 500] = np.nan
        # write_img(self, filename, im_proj, im_geotrans, im_data)
        tiff.write_img(output_tif, im_proj, im_geotrans, result)

    def start(self):
        """
        程序执行逻辑
            先执行参数检查、再执行核心处理逻辑
        """
        self.check_params()
        if os.path.exists(self._pm_path):
            self.model_build()

    @classmethod
    def run(cls):
        """
        算法程序执行函数，需对逻辑进行异常捕获

        处理逻辑：
            1. 从程序调用命令行获取参数xml文件
            2. 创建参数解析类，解析xml参数
            3. 获取xml参数，创建算法类实例
            4. 依次调用类实例的 参数验证(check_params) 及 处理逻辑(process) 方法

        调用方式:
            python.exe demo_alg_python.py E:/datamining/params.xml
        """

        # 算法参数定义
        ParamName_spatial_name = 'spatial_name'
        ParamName_time = 'time'
        ParamName_Sentinel2_path = 'Sentinel2_path'
        ParamName_Tem_path = 'Tem_path'
        ParamName_Rainfall_path = 'Rainfall_path'
        ParamName_Wind_path = 'Wind_path'
        ParamName_PM_Station_path = 'PM_Station_path'
        ParamName_inmodel_idx = 'inmodel_idx'
        ParamName_train_size = 'train_size'


        try:
            # 从命令行获取参数
            if len(sys.argv) == 0:
                print('参数文件为空，请重试')
                exit(3)
            #xml_path = r'D:/deploy/work/datamining/316b83d6da1c41afb58394ffe01fc81b/XCO2_Project/XCO2_Project.xml'
            #arg_parser = ArgParser(xml_path)
            arg_parser = ArgParser(sys.argv[1])
            params = arg_parser.parse()
            param_spatial_name = params.get(ParamName_spatial_name)
            param_time = params.get(ParamName_time)
            param_Sentinel2_path = params.get(ParamName_Sentinel2_path)
            param_Tem_path = params.get(ParamName_Tem_path)
            param_Rainfall_path = params.get(ParamName_Rainfall_path)
            param_Wind_path = params.get(ParamName_Wind_path)
            param_PM_Station_path = params.get(ParamName_PM_Station_path)
            param_inmodel_idx = params.get(ParamName_inmodel_idx)
            param_train_size = params.get(ParamName_train_size)


        except Exception as error:
            print('程序执行出现异常，详细原因为：\n{0}'.format(error))
            sys.exit(3)

        param_Model = 'PM2d5'
        xls_path = param_PM_Station_path.replace("\\", '\\\\')
        #根据日期新建一个输出文件夹

        #root_dir= os.path.dirname(xls_path)
        root_dir = 'D:\\miyun\\03_Result\\'+param_spatial_name+'_'+str(param_time)
        # print(file_path, root_dir)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)


        pm_extract_xls = os.path.join(root_dir,
                                      '_'.join([param_spatial_name, 'pm2d5', str(param_time), 'extract.xlsx']))
        pm_valid_xls = os.path.join(root_dir,
                                      '_'.join([param_spatial_name, 'pm2d5',  str(param_time), 'valid.xlsx']))
        pm_model_record_xls = os.path.join(root_dir,
                                      '_'.join([param_spatial_name, 'pm2d5',  str(param_time), 'record.xlsx']))
        PM_Output = os.path.join(root_dir,
                                      '_'.join([param_spatial_name, 'pm2d5',  str(param_time)+'.tif']))
        XCO2_Output = os.path.join(root_dir,
                                      '_'.join([param_spatial_name, 'XCO2',  str(param_time)+'.tif']))

        # pm_valid_xls = root_dir + '\pm2d5_' + param_spatial_name + '_' + param_time + '_valid.xlsx'
        # pm_model_record_xls = root_dir + '\pm2d5_' + param_spatial_name + '_' + param_time + '_record.xlsx'
        # PM_Output = root_dir + '\pm2d5_' + param_spatial_name + '_' + param_time + '.tif'
        # XCO2_Output = root_dir + '\XCO2_' + param_spatial_name + '_' + param_time + '.tif'
        # 波段提取和指数计算
        BandData_Extract_math_m1 = BandData_Extract_math(param_PM_Station_path,
                                                         param_Sentinel2_path,
                                                         param_Tem_path,
                                                         param_Wind_path,
                                                         pm_extract_xls)
        BandData_Extract_math_m1.model_build()
        # PM2.5模型构建
        LinearModel_build_m1 = LinearModel_build(param_Model, pm_extract_xls, param_inmodel_idx, param_train_size,
                                                 pm_valid_xls, pm_model_record_xls)
        LinearModel_build_m1.model_build()

        #  PM2.5模型应用
        PMLinearModel_apply_m1 = PMLinearModel_apply(pm_model_record_xls, param_Sentinel2_path,
                                                     param_Tem_path, param_Wind_path, PM_Output)
        PMLinearModel_apply_m1.model_build()

        # XCO2模型应用
        obj = cls(PM_Output, param_Tem_path, param_Rainfall_path, XCO2_Output)
        obj.start()


if __name__ == '__main__':
    CO2_Model_apply.run()
