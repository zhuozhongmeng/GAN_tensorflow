import os
import csv
import numpy as np
import cv2
import tensorflow as tf
import random


# ---------------------------------系统设定用的函数汇总-----------------------------------------------


# 读取当前项目状态的设定参数
def read_setting_csv(file_name):
    data_last = 0
    f = open(file_name, "r")
    data = csv.reader(f)
    for data_i in data:
        data_last = data_i
    return data_last  # 只返回最新的


# 保存当前项目状态的设定参数
def save_setting_csv(data, file_name):
    if not os.path.exists(file_name):
        print("文件不存在，为您创建文件成功")
        file = open(file_name, 'a+', newline='')
        writer = csv.writer(file)
        writer.writerow(['is_reload', 'rate', 'rate_speed', 'trained_times', 'imgs'])  # 写入csv文件的表头
        writer.writerow(data)
        file.close
        print("保存文件成功")
    else:
        file = open(file_name, 'a+', newline='')
        writer = csv.writer(file)
        writer.writerow(data)
        file.close
        print("保存文件成功")


# ---------------------------------系统设定用的函数汇总-----------------------------------------------
# ---------------------------------文件读取及文件存储的函数汇总---------------------------------------

# 读取CSV中的数据
def read_csv(files_name):
    f = open(files_name, "r", errors='ignore')
    data = csv.reader(f)
    data_test = []
    for i in data:
        data_test.append(i)
    data_test = np.stack(data_test, axis=0)
    data_test = data_test[1:]
    random_labels_test = data_test.astype("float")
    return random_labels_test


# 根据需求创建对应文件夹
def set_path(path, dir_list):
    if not os.path.exists(path):
        for i in range(len(dir_list)):
            os.makedirs(path + "/" + dir_list[i])
        print("创建文件夹完成")
    for i in range(len(dir_list)):
        if not os.path.exists(path + "/" + dir_list[i]):
            os.makedirs(path + "/" + dir_list[i])
            print("补充文件夹", dir_list[i])


# 获取文件夹中文件数量，返回所有文件名+文件总数
def get_dir(path):
    print("start read files")
    dir_files_list = os.listdir(path)
    print("end")
    return dir_files_list, len(dir_files_list)


# 保存测试展示用的随机变量，用于记录生成器的训练效果过程
def load_test_sample(file_name):
    data = np.loadtxt(file_name)
    print("now", data)
    return data


# 读取CSV数据库文件，返回整个文件的全部数据，不做任何筛选处理，文件格式依然是STR
def load_img_index(file_name, is_stack=False):
    file = open(file_name, "r", errors='ignore')
    data_list = []
    data = csv.reader(file)
    for data_i in data:
        data = data_i[0].split()
        data_list.append(data)
    if is_stack:
        data_n = np.vstack(data_list)  # 返回的是numpy数组
    else:
        data_n = data_list  # 返回的是list
    return data_n


def get_sample(data, batch_size):
    data = random.sample(data, batch_size)
    data = np.vstack(data)
    return data


def load_rate(filename="setting.txt"):
    data = np.loadtxt(filename)
    is_reload = int(data[0])
    rate = float(data[1])
    speed = float(data[2])
    trained_times = int(data[3])
    now_imgs = int(data[4])
    return is_reload, rate, speed, trained_times, now_imgs


def save_rate(is_reload, rate, speed, trained_times, now_imgs, file_name="setting.txt"):
    rate_data = []
    rate_data.append(is_reload)
    rate_data.append(rate)
    rate_data.append(speed)
    rate_data.append(trained_times)
    rate_data.append(now_imgs)
    np.savetxt(file_name, rate_data)


# 读取图片文件 从提供的list中全部提取


def get_image(path, images_list):
    images = []
    for img_list in images_list:
        image = cv2.imread(path + "/" + img_list)
        image = cv2.resize(image[20:198, 0:178], (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        images.append(image)
    return images


# 保存生成的图像，使用cv2
def save_img(samples, imgs, path):
    for q in range(np.shape(samples)[0]):
        image = cv2.cvtColor(samples[q], cv2.COLOR_BGR2RGB)
        # image = samples[q]

        cv2.imwrite(path + '/images/' + str(imgs) + "_" + str(q) + '.jpg', image * 255)


# ---------------------------------文件读取及文件存储的函数汇总-----------------------------------------------
# ---------------------------------(TF)相关的函数汇总---------------------------------------------------


def m_conv2d(img, filters=4, strides=2, padding="SAME", in_shape=[2, 2, 2], out_shape=[2, 2, 2], is_relu=True,
             is_bn=True, is_train=True, name_header="M"):
    name_header = str(name_header)
    g_w = tf.compat.v1.get_variable(name=name_header + '-w',
                                    initializer=tf.random.normal([filters, filters, in_shape[2], out_shape[2]],
                                                                 stddev=0.02))
    g_b = tf.compat.v1.get_variable(name=name_header + '-b', initializer=tf.zeros(out_shape[2]))
    img = tf.nn.conv2d(img, g_w, [1, strides, strides, 1], padding=padding, name=name_header)
    if is_bn:
        img = tf.layers.batch_normalization(img, training=is_train, name=name_header + "_bn")
    if is_relu:
        img = tf.nn.leaky_relu(img + g_b, alpha=0.2, name=name_header + "_relu")
    print(img)
    return img


def m_conv2d_transpose(img, filters=4, strides=2, padding="SAME", in_shape=[2, 2, 2], out_shape=[2, 2, 2], is_relu=True,
                       is_bn=True, is_train=True, name_header="M"):
    img_shape = tf.shape(img)
    name_header = str(name_header)
    g_w = tf.compat.v1.get_variable(name=name_header + '-w',
                                    initializer=tf.random.normal([filters, filters, out_shape[2], in_shape[2]],
                                                                 stddev=0.02))
    g_b = tf.compat.v1.get_variable(name=name_header + '-b', initializer=tf.zeros(out_shape[2]))
    img = tf.nn.conv2d_transpose(img, g_w, [img_shape[0], out_shape[0], out_shape[1], out_shape[2]],
                                 [1, strides, strides, 1], padding=padding, name=name_header)
    if is_bn:
        img = tf.layers.batch_normalization(img, training=is_train, name=name_header + "_bn")
    if is_relu:
        img = tf.nn.relu(img + g_b, name=name_header + "_relu")
    print(img)
    return img


# ---------------------------------(TF)相关的函数汇总---------------------------------------------------

if __name__ == '__main__':
    print("你不应该在函数库文件中做运行，请找到对应主文件，谢谢")
