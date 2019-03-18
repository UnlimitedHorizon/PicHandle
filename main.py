"""XXXX"""
# -*- coding: utf-8 -*-
# a
# 这个demo里，用到了正则表达式库re（标准库）、文件及路径操作库os（标准库）、矩阵运算库numpy（附加库需下载）、图片视频处理库opencv（附加库需下载）
import re
import os
import numpy as np
import cv2

# path = os.getcwd()
path = os.path.dirname(__file__)

class Data:
    '''class Data'''
    def __init__(self):
        self.picture_width = -1
        self.picture_height = -1
        self.canvas_width = -1
        self.canvas_height = -1
        self.coordinate_array_original_np = np.zeros([1, 2], dtype=np.int)
        self.coordinate_array_normalized_np = np.zeros([1, 2], dtype=np.float64)
        self.coordinate_array_draw_np = np.zeros([1, 2], dtype=np.int)
        self.gray_list = []
        self.is_pu_list = []
        self.coordinate_list = []
        self.read_from_file(path + "/input/pic.plt")


    def __set_picture_size(self, width, height):
        self.picture_width = width
        self.picture_height = height


    def __set_canvas_size(self, width, height):
        self.canvas_width = width
        self.canvas_height = height


    def set_size(self, pic_width, pic_height, canvas_width, canvas_height):
        '''set picture and canvas size'''
        self.__set_picture_size(pic_width, pic_height)
        self.__set_canvas_size(canvas_width, canvas_height)
        bias = np.asarray(
            [(int)((self.canvas_width - self.picture_width)/2),
             (int)((self.canvas_height - self.picture_height)/2)], dtype=np.int)
        picture_size = np.asarray([self.picture_width, self.picture_height], dtype=np.int)
        self.coordinate_array_draw_np = (
            (self.coordinate_array_normalized_np*picture_size).astype(np.int)
            + bias)


    def add_datum(self, gray, is_pu, coor_x, coor_y):
        '''add datum'''
        self.gray_list.append(gray)
        self.is_pu_list.append(is_pu)
        self.coordinate_list.append([coor_x, coor_y])


    def read_from_file(self, file_path):
        '''read_from_file'''
        data_plt_file = open(file_path, "r")
        re_gray = re.compile(r"SP(?P<gray>\d+);")
        re_coor = re.compile(r"P(?P<ud_flag>[UD])(?P<coor_x>-?\d+)\s(?P<coor_y>-?\d+);")
        gray = 0

        for line in data_plt_file:
            if re_gray.match(line):
                gray = (int)(re_gray.match(line).group("gray"))
            elif re_coor.match(line):
                match_result = re_coor.match(line)
                coor_x = (int)(match_result.group("coor_x"))
                coor_y = (int)(match_result.group("coor_y"))
                is_pu = (match_result.group("ud_flag") == "U")
                self.add_datum(gray, is_pu, coor_x, coor_y)
        data_plt_file.close()
        self.normalize()


    def normalize(self):
        '''normalize'''
        self.coordinate_array_original_np = np.asarray(self.coordinate_list, dtype=np.int)
        coordinate_min_np = np.amin(self.coordinate_array_original_np, 0)
        # coordinate_max_np = np.amax(self.coordinate_array_original_np, 0)
        coordinate_ptp_np = np.ptp(self.coordinate_array_original_np, 0)
        self.picture_width = coordinate_ptp_np[0]
        self.picture_height = coordinate_ptp_np[1]
        self.coordinate_array_normalized_np = (np.ones([1, 2], dtype=np.float64) -
                                               ((self.coordinate_array_original_np
                                                 - coordinate_min_np).astype(np.float64))
                                               / (coordinate_ptp_np.astype(np.float64)))


    def output_data(self):
        '''output data to file'''
        outfile = open(path + "/output/plt_extract.json", "w")
        outfile.write("{\n\t\"data\": [\n")
        size = (
            (int)(self.coordinate_array_normalized_np.size
                  /self.coordinate_array_normalized_np.ndim))
        for i in range(0, size):
            outfile.write("\t\t{\n")
            outfile.write("\t\t\t\"gray\": {gray},\n".format(gray=self.gray_list[i]))
            if self.is_pu_list[i]:
                outfile.write("\t\t\t\"pu_pd_flag\": \"pu\",\n")
            else:
                outfile.write("\t\t\t\"pu_pd_flag\": \"pd\",\n")
            outfile.write("\t\t\t\"x\": {x},\n".format(x=self.coordinate_array_normalized_np[i, 0]))
            outfile.write("\t\t\t\"y\": {y}\n".format(y=self.coordinate_array_normalized_np[i, 1]))
            if i < (size - 1):
                outfile.write("\t\t},\n")
            else:
                outfile.write("\t\t}\n")
        outfile.write("\t]\n}")
        outfile.close()

    def output_coordinate(self):
        '''output drawing coordinate'''
        bias = np.asarray(
            [(int)((self.canvas_width - self.picture_width)/2),
             (int)((self.canvas_height - self.picture_height)/2)], dtype=np.int)
        picture_size = np.asarray([self.picture_width, self.picture_height], dtype=np.int)
        coordinate_array_draw_np = (
            (self.coordinate_array_normalized_np*picture_size).astype(np.int)
            + bias)
        outfile = open(path + "/output/plt_draw_extract.json", "w")
        outfile.write("{\n\t\"data\": [\n")
        size = (
            (int)(coordinate_array_draw_np.size
                  /coordinate_array_draw_np.ndim))
        for i in range(0, size):
            outfile.write("\t\t{\n")
            outfile.write("\t\t\t\"gray\": {gray},\n".format(gray=self.gray_list[i]))
            if self.is_pu_list[i]:
                outfile.write("\t\t\t\"pu_pd_flag\": \"pu\",\n")
            else:
                outfile.write("\t\t\t\"pu_pd_flag\": \"pd\",\n")
            outfile.write("\t\t\t\"x\": {x},\n".format(x=coordinate_array_draw_np[i, 0]))
            outfile.write("\t\t\t\"y\": {y}\n".format(y=coordinate_array_draw_np[i, 1]))
            if i < (size - 1):
                outfile.write("\t\t},\n")
            else:
                outfile.write("\t\t}\n")
        outfile.write("\t]\n}")
        outfile.close()


class DrawingBoard:
    '''class to draw'''
    def __init__(self):
        self.data = Data()
        self.data.set_size(700, 850, 720, 1080)
        self.data.output_data()
        self.data.output_coordinate()
        self.output_picture = np.ones((self.data.canvas_height, self.data.canvas_width))*255

    def draw_line(self):
        '''draw line'''
        self.output_picture = np.ones((self.data.canvas_height, self.data.canvas_width))*255
        size = (
            (int)(self.data.coordinate_array_draw_np.size
                  /self.data.coordinate_array_draw_np.ndim))
        last_coor = None
        for i in range(0, size):
            if not self.data.is_pu_list[i]:
                gray_grade = 9
                max_grade = 256
                # gray = self.data.gray_list[i]
                gray = 8
                gray = (int)(((gray_grade - 1 - gray)/gray_grade)*max_grade)

                if last_coor is not None:
                    self.__draw_line(last_coor, self.data.coordinate_array_draw_np[i], gray)
                last_coor = self.data.coordinate_array_draw_np[i]
            else:
                last_coor = None
        cv2.imshow("line", self.output_picture.astype("uint8"))
        cv2.imwrite("output/line_graph.png", self.output_picture)

    def draw_point(self):
        '''draw point'''
        self.output_picture = np.ones((self.data.canvas_height, self.data.canvas_width))*255
        size = (
            (int)(self.data.coordinate_array_draw_np.size
                  /self.data.coordinate_array_draw_np.ndim))
        for i in range(0, size):
            if not self.data.is_pu_list[i]:
                gray_grade = 9
                max_grade = 256
                # gray = self.data.gray_list[i]
                gray = 8
                gray = (int)(((gray_grade - 1 - gray)/gray_grade)*max_grade)

                self.output_picture[self.data.coordinate_array_draw_np[i, 1],
                                    self.data.coordinate_array_draw_np[i, 0]] = gray
        cv2.imshow("point", self.output_picture.astype("uint8"))
        cv2.imwrite("output/point_graph.png", self.output_picture)

    def draw_line_serial(self):
        '''draw line'''
        self.output_picture = np.ones((self.data.canvas_height, self.data.canvas_width))*255
        size = (
            (int)(self.data.coordinate_array_draw_np.size
                  /self.data.coordinate_array_draw_np.ndim))

        last_coor = None
        pic_num = 0
        for i in range(0, size):
            if not self.data.is_pu_list[i]:
                gray_grade = 9
                max_grade = 256
                # gray = self.data.gray_list[i]
                gray = 8
                gray = (int)(((gray_grade - 1 - gray)/gray_grade)*max_grade)

                if last_coor is not None:
                    self.__draw_line(last_coor, self.data.coordinate_array_draw_np[i], gray)
                last_coor = self.data.coordinate_array_draw_np[i]
            else:
                cv2.imwrite("output/serial/line_serial{:0>3d}.png".format(pic_num), self.output_picture)
                pic_num += 1
                last_coor = None
        cv2.imwrite("output/serial/line_serial{:0>3d}.png".format(pic_num), self.output_picture)
        pic_num += 1
        cv2.imshow("line", self.output_picture.astype("uint8"))
        cv2.imwrite("output/line_graph.png", self.output_picture)
        self.pic_to_video(pic_num)


    def pic_to_video(self, serial_num):
        '''pic_to_video'''
        # 为保存视频做准备
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # 第三个参数则是镜头快慢的，20为正常，小于二十为慢镜头
        out_video = cv2.VideoWriter('./output/output.avi',
                                    fourcc, 2.0,
                                    (self.data.canvas_width, self.data.canvas_height))
        for i in range(0, serial_num):
            frame = cv2.imread("output/serial/line_serial{:0>3d}.png".format(i))
            out_video.write(frame)


    def __draw_line(self, coor_start, coor_end, color):
        x_0 = coor_start[0]
        x_1 = coor_end[0]
        if x_0 > x_1:
            x_0, x_1 = x_1, x_0
            y_0 = coor_end[1]
            y_1 = coor_start[1]
        else:
            y_0 = coor_start[1]
            y_1 = coor_end[1]

        if x_0 == x_1:
            if y_0 > y_1:
                y_0, y_1 = y_1, y_0
            coor_x = x_0
            for coor_y in range(y_0, y_1+1):
                self.output_picture[coor_y, coor_x] = color
        else:
            delta_x = x_1 - x_0
            delta_y = y_1 - y_0
            k = delta_y/delta_x
            if k > 0.5:
                coor_x = x_0
                step = 1/k
                bias = -0.5
                for coor_y in range(y_0, y_1+1):
                    self.output_picture[coor_y, coor_x] = color
                    bias += step
                    if bias >= 0:
                        coor_x += 1
                        bias -= 1
            elif k > 0:
                coor_y = y_0
                step = k
                bias = -0.5
                for coor_x in range(x_0, x_1+1):
                    self.output_picture[coor_y, coor_x] = color
                    bias += step
                    if bias >= 0:
                        coor_y += 1
                        bias -= 1
            elif k > -0.5:
                coor_y = y_0
                step = -k
                bias = -0.5
                for coor_x in range(x_0, x_1+1):
                    self.output_picture[coor_y, coor_x] = color
                    bias += k
                    if bias >= 0:
                        coor_y -= 1
                        bias -= 1
            else:
                coor_x = x_0
                step = -1/k
                bias = -0.5
                for coor_y in range(y_0, y_1-1, -1):
                    self.output_picture[coor_y, coor_x] = color
                    bias += step
                    if bias >= 0:
                        coor_x += 1
                        bias -= 1


Board = DrawingBoard()
Board.draw_line_serial()
Board.draw_point()
cv2.waitKey()

# def fixLight():
#     '''fix Light'''
#     picture = cv2.imread("input/water.bmp")
#     row_num, col_num, channel_num = picture.shape
#     picture_gray = np.zeros((row_num, col_num, channel_num))
#     for r in range(row_num):
#         for l in range(col_num):
#             R = picture[r,l,0]
#             G = picture[r,l,1]
#             B = picture[r,l,2]
#             Y = round( 0.256788 * R + 0.504129 * G + 0.097906 * B) +  16 
#             U = round(-0.148223 * R - 0.290993 * G + 0.439216 * B) + 128
#             V = round( 0.439216 * R - 0.367788 * G - 0.071427 * B) + 128
#             Y -= 60
#             picture_gray[r,l,0] =  Y + 1.4075 * (V - 128)
#             picture_gray[r,l,1] =  Y - 0.3455 * (U - 128) - (0.7169 * (V - 128))
#             picture_gray[r,l,2] =  Y + 1.7790 * (U - 128)
#             # for c in range(channel_num):
#             #     R = Y + 1.4075 * (V - 128)
#             #     G = Y - 0.3455 * (U - 128) - (0.7169 * (V - 128))
#             #     B = Y + 1.7790 * (U - 128)
#             #     picture_gray[r, l, c] = picture[r, l, c]

#     cv2.imshow("picture_gray", picture_gray.astype("uint8"))
#     cv2.imwrite("output/water.bmp", picture_gray)
# fixLight()
# cv2.waitKey()


# def main():
#     '''main function'''
#     picture = cv2.imread("input/test.png")
#     row_num, col_num, channel_num = picture.shape
#     picture_gray = np.zeros((row_num, col_num))
#     for r in range(row_num):
#         for l in range(col_num):
#             gray = 1 / 3 * picture[r, l, 0] + 1 / 3 * picture[r, l, 1] + 1 / 3 * picture[r, l, 2]
#             grade = 8
#             gray = (int((gray+1)/(256/grade)))*grade-1
#             picture_gray[r, l] = gray

#     cv2.imshow("picture_gray", picture_gray.astype("uint8"))
#     cv2.imwrite("output/gray.png", picture_gray)
#     cv2.waitKey()

# def pltToImg():
#     '''plt to image'''
#     file_plt = open("input/pic.plt")


# main()
