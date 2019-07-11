import tensorflow as tf
from bata01 import functions as m_fc
import numpy as np
import os


class CGAN:
    def __init__(self, name="Cclass", batch_size="100", path="project"):
        self.name = name
        self.PATH = str(path)  # 项目保存的文件夹
        self.DIR = ["images", "sum"]  # 项目的所需的文件夹
        self.BATCH_SIZE = batch_size
        self.labels_SIZE = 40
        self.RANDOM_SAMPLES_SIZE = 344
        # 设定可变网络参数数组
        self.g_val_list = []
        self.d_val_list = []
        # 首先先设定placeholder
        self.reuse = tf.compat.v1.placeholder(dtype=bool)
        self.rate = tf.compat.v1.placeholder(dtype=float)
        self.is_train = tf.compat.v1.placeholder(dtype=bool)
        self.is_relu = tf.compat.v1.placeholder(dtype=bool)
        self.is_bn = tf.compat.v1.placeholder(dtype=bool)
        self.labels = tf.compat.v1.placeholder(dtype=float, shape=[None, self.labels_SIZE])
        self.labels_g = tf.compat.v1.placeholder(dtype=float, shape=[None, self.labels_SIZE])
        self.labels_d = tf.compat.v1.placeholder(dtype=float, shape=[None, self.labels_SIZE])
        self.random_samples = tf.compat.v1.placeholder(dtype=float, shape=[None, self.RANDOM_SAMPLES_SIZE])
        self.images_input = tf.compat.v1.placeholder(dtype=float, shape=[None, 128, 128, 3])
        # 先定义三个损失函数，留给以后优化
        self.d_loss = 0
        self.g_loss = 0
        self.creat_net()
        # 这里要研究下怎么为一个对象创建一个session
        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 1  # 占用GPU90%的显存
        self.config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=self.config)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        # 保存模型
        self.saver = tf.compat.v1.train.Saver()
        print("你好 我是你创建的一个类", name)
        print("首先读取或者创建文件夹")
        m_fc.set_path(self.PATH, self.DIR)
        if not os.path.exists(self.PATH + '/model/'):
            print("本次为全新训练，无历史模型")
        else:
            print("开始读取模型")
            self.saver.restore(self.sess, self.PATH + '/model/model.ckpt')
            print("读取模型结束")
        # tensorboard 相关
        self.merge = tf.compat.v1.summary.merge_all()
        self.sum_writer = tf.compat.v1.summary.FileWriter(self.PATH + "/sum/", self.sess.graph)

        # <<进行对象数据的初始化检测----------------------------------------

        # ------------------------------------------end>>

    # <<创建需要的多个神经网络------------------------------------------
    def g_net(self, labels_g, random_samples, is_reuse=False, is_train=False):
        with tf.compat.v1.variable_scope(self.name + "G_NET", reuse=is_reuse):
            labels_g_chanels = tf.reshape(labels_g, [-1, 1, 1, 40])
            print(labels_g_chanels)
            batch_size = tf.shape(labels_g)[0]
            input = tf.compat.v1.concat([random_samples, labels_g], 1)
            input = tf.reshape(input, [-1, 1, 1, self.labels_SIZE + self.RANDOM_SAMPLES_SIZE], "input_x")
            g0 = m_fc.m_conv2d_transpose(input, 4, 1, "VALID", [1, 1, 384], [4, 4, 384], is_relu=True, is_bn=True,
                                         is_train=is_train, name_header="g0")
            g0_labels = tf.tile(labels_g_chanels, [1, 4, 4, 1])
            g0 = tf.compat.v1.concat([g0, g0_labels], axis=3)
            g1 = m_fc.m_conv2d_transpose(g0, 4, 2, "SAME", [4, 4, 424], [8, 8, 192], is_relu=True, is_bn=True,
                                         is_train=is_train, name_header="g1")
            g1_labels = tf.tile(labels_g_chanels, [1, 8, 8, 1])
            g1 = tf.compat.v1.concat([g1, g1_labels], axis=3)
            g2 = m_fc.m_conv2d_transpose(g1, 4, 2, "SAME", [8, 8, 232], [16, 16, 96], is_relu=True, is_bn=True,
                                         is_train=is_train, name_header="g2")
            g2_labels = tf.tile(labels_g_chanels, [1, 16, 16, 1])
            g2 = tf.compat.v1.concat([g2, g2_labels], axis=3)
            g3 = m_fc.m_conv2d_transpose(g2, 4, 2, "SAME", [16, 16, 136], [32, 32, 48], is_relu=True, is_bn=True,
                                         is_train=is_train, name_header="g3")
            g3_labels = tf.tile(labels_g_chanels, [1, 32, 32, 1])
            g3 = tf.compat.v1.concat([g3, g3_labels], axis=3)
            g4 = m_fc.m_conv2d_transpose(g3, 4, 2, "SAME", [32, 32, 88], [64, 64, 24], is_relu=True, is_bn=True,
                                         is_train=is_train, name_header="g4")
            g4_labels = tf.tile(labels_g_chanels, [1, 64, 64, 1])
            g4 = tf.compat.v1.concat([g4, g4_labels], axis=3)
            g5 = m_fc.m_conv2d_transpose(g4, 4, 2, "SAME", [64, 64, 64], [128, 128, 12], is_relu=True, is_bn=True,
                                         is_train=is_train, name_header="g5")
            g5_labels = tf.tile(labels_g_chanels, [1, 128, 128, 1])
            g5 = tf.compat.v1.concat([g5, g5_labels], axis=3)
            g6 = m_fc.m_conv2d_transpose(g5, 1, 1, "SAME", [128, 128, 52], [128, 128, 3], is_relu=False, is_bn=False,
                                         is_train=is_train, name_header="g6")
            out = (tf.nn.tanh(g6) + 1) / 2
            # self.summary_g_out_images = tf.compat.v1.summary.image("g_out_images", out, max_outputs=4)
            # self.summary_g0 = tf.compat.v1.summary.histogram("g0", g0)
            # self.summary_g1 = tf.compat.v1.summary.histogram("g1", g1)
            # self.summary_g2 = tf.compat.v1.summary.histogram("g2", g2)
            # self.summary_g3 = tf.compat.v1.summary.histogram("g3", g3)
            # self.summary_g4 = tf.compat.v1.summary.histogram("g4", g4)
            # self.summary_g5 = tf.compat.v1.summary.histogram("g5", g5)
            # self.summary_g6 = tf.compat.v1.summary.histogram("g6", g6)
            # self.summary_g_out = tf.compat.v1.summary.histogram("g_out", out)
            print("这里创建一个生成器网络", out)
            g_val_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope=self.name + "G_NET")
            return out, g_val_list

    def d_net(self, data, labels, is_reuse=False, is_train=False):
        with tf.compat.v1.variable_scope(self.name + "D_NET", reuse=is_reuse):
            # 对标签进行处理
            labels= tf.reshape(labels * 0.5 + 0.5, [-1, 1, 1, 40])
            print(labels)
            labels = tf.tile(labels, [1, 128, 128, 1])
            print(labels)
            d_in = tf.compat.v1.concat([data, labels], axis=3)
            print(d_in)
            d0 = m_fc.m_conv2d(d_in, 4, 1, "SAME", [128, 128, 43], [128, 128, 52], is_relu=True, is_bn=False,
                               is_train=is_train, name_header="d0")
            d1 = m_fc.m_conv2d(d0, 4, 2, "SAME", [128, 128, 52], [64, 64, 64], is_relu=True, is_bn=False,
                               is_train=is_train, name_header="d1")
            d2 = m_fc.m_conv2d(d1, 4, 2, "SAME", [64, 64, 64], [32, 32, 88], is_relu=True, is_bn=False,
                               is_train=is_train, name_header="d2")
            d3 = m_fc.m_conv2d(d2, 4, 2, "SAME", [32, 32, 88], [16, 16, 136], is_relu=True, is_bn=False,
                               is_train=is_train, name_header="d3")
            d4 = m_fc.m_conv2d(d3, 4, 2, "SAME", [16, 16, 136], [8, 8, 232], is_relu=True, is_bn=False,
                               is_train=is_train, name_header="d4")
            d5 = m_fc.m_conv2d(d4, 4, 2, "SAME", [8, 8, 232], [4, 4, 424], is_relu=True, is_bn=False,
                               is_train=is_train, name_header="d5")
            d6 = m_fc.m_conv2d(d5, 4, 1, "VALID", [4, 4, 424], [1, 1, 424], is_relu=False, is_bn=False,
                               is_train=is_train, name_header="d6")
            d_labels = m_fc.m_conv2d(d6, 1, 1, "VALID", [1, 1, 424], [1, 1, 40], is_relu=False, is_bn=False,
                                     is_train=is_train, name_header="d_labels")
            dout_logit = tf.reshape(d6, [-1, 424])
            dout_labels = tf.reshape(d_labels, [-1, 40])
            d_val_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope=self.name + "D_NET")
            print("这里创建一个仿真判断器网络")
            # self.summary_images_in = tf.compat.v1.summary.image("images_esp", data, max_outputs=32)
            # self.summary_images_labels = tf.compat.v1.summary.image("images_label", labels_the_channel, max_outputs=32)
            # self.summary_d0 = tf.compat.v1.summary.histogram("d0", d0)
            # self.summary_d1 = tf.compat.v1.summary.histogram("d1", d1)
            # self.summary_d2 = tf.compat.v1.summary.histogram("d2", d2)
            # self.summary_d3 = tf.compat.v1.summary.histogram("d3", d3)
            # self.summary_d4 = tf.compat.v1.summary.histogram("d4", d4)
            # self.summary_d5 = tf.compat.v1.summary.histogram("d5", d5)
            # self.summary_d6 = tf.compat.v1.summary.histogram("d6", d6)
            # self.summary_losit = tf.compat.v1.summary.histogram("logist", dout_logit)
            # self.summary_labels = tf.compat.v1.summary.histogram("labels", dout_labels)
            return dout_logit, dout_labels, d_val_list

    def creat_net(self):
        self.g_images, self.g_val_list = self.g_net(labels_g=self.labels_g, random_samples=self.random_samples,
                                                    is_reuse=False, is_train=False)
        self.dout_logit, self.dout_labels, self.d_val_list = self.d_net(data=self.images_input, labels=self.labels,
                                                                        is_reuse=False, is_train=False)
        self.dout_logit_g, self.dout_labels_g, _ = self.d_net(data=self.g_images, labels=self.labels_g,
                                                              is_reuse=True, is_train=False)
        print("创建网络完成")
        self.d_labels_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dout_labels, labels=self.labels * 0.5 + 0.5))
        self.d_labels_g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dout_labels_g, labels=self.labels_g * 0.5 + 0.5))

        self.d_loss = tf.reduce_mean(self.dout_logit_g) - tf.reduce_mean(
            self.dout_logit) + self.get_gp_item(images=self.images_input,
                                                g_images=self.g_images, labels_g=self.labels_g,
                                                labels=self.labels)
        self.d_loss = self.d_labels_loss + tf.reduce_mean(self.dout_logit_g) - tf.reduce_mean(
            self.dout_logit) + self.get_gp_item(images=self.images_input,
                                                g_images=self.g_images, labels_g=self.labels_g, labels=self.labels)
        self.g_loss = self.d_labels_g_loss - tf.reduce_mean(self.dout_logit_g)
        self.g_admin = tf.compat.v1.train.AdamOptimizer(self.rate, beta1=0.5, beta2=0.9).minimize(self.g_loss,
                                                                                                  var_list=self.g_val_list)  # 生成器真实性训练
        self.d_admin = tf.compat.v1.train.AdamOptimizer(self.rate, beta1=0.5, beta2=0.9).minimize(self.d_loss,
                                                                                                  var_list=self.d_val_list)  # 判别器分辨力训练

        self.summary_g_loss = tf.compat.v1.summary.scalar("0_g_loss", self.g_loss)
        self.summary_d_loss = tf.compat.v1.summary.scalar("0_d_loss", self.d_loss)
        self.summary_d_l_loss = tf.compat.v1.summary.scalar("0_d_l_loss", self.d_labels_loss)
        self.summary_d_g_loss = tf.compat.v1.summary.scalar("0_d_g_loss", self.d_labels_g_loss)

    def train_net(self, images_input, random_samples, labels, labels_g, rate, times):
        for i in range(20):
            self.sess.run(self.g_admin,
                          feed_dict={self.labels: labels, self.random_samples: random_samples,
                                     self.rate: rate,
                                     self.images_input: images_input, self.labels_g: labels_g,
                                     self.is_train: True, self.reuse: True})
            self.sess.run(self.g_admin,
                          feed_dict={self.labels: labels, self.random_samples: random_samples,
                                     self.rate: rate,
                                     self.images_input: images_input, self.labels_g: labels_g,
                                     self.is_train: True, self.reuse: True})
            self.sess.run(self.g_admin,
                          feed_dict={self.labels: labels, self.random_samples: random_samples,
                                     self.rate: rate,
                                     self.images_input: images_input, self.labels_g: labels_g,
                                     self.is_train: True, self.reuse: True})
            self.sess.run(self.d_admin,
                          feed_dict={self.labels: labels, self.random_samples: random_samples,
                                     self.rate: rate,
                                     self.images_input: images_input, self.labels_g: labels_g,
                                     self.is_train: True, self.reuse: True})

        if times % 10 == 0:  # 每10轮保存一次模型，也就是运算200次
            self.saver.save(self.sess, self.PATH + '/model/model.ckpt')

    def test_net(self, images_input, labels, random_sample, times, labels_g, ):
        d_loss_now, g_loss_now, g_images_now = self.sess.run(
            [self.d_loss, self.g_loss, self.g_images],
            feed_dict={self.images_input: images_input,
                       self.labels: labels,
                       self.random_samples: random_sample, self.labels_g: labels_g, self.is_train: False,
                       self.reuse: True})

        self.the_summary = self.sess.run(self.merge,
                                         feed_dict={self.labels: labels, self.images_input: images_input,
                                                    self.random_samples: random_sample, self.labels_g: labels_g,
                                                    self.is_train: False, self.reuse: True})
        self.sum_writer.add_summary(self.the_summary, times)
        return d_loss_now, g_loss_now, g_images_now

    # --------------------------------------------------------------end>>
    # <<一些优化函数----------------------------------------------------
    #
    def get_gp_item(self, images, g_images, labels_g, labels):
        eps = tf.random.uniform([tf.shape(images)[0], 128, 128, 3], minval=0., maxval=1.)  # 就是建一个和图像一致的随机数
        eps_1 = tf.random.uniform([tf.shape(labels)[0], 40], minval=0., maxval=1.)  # 就是建一个和图像一致的随机数
        input_with_random = eps * images + (1 - eps) * g_images  # 混合了真是样本和生成样本之后的混合样本
        labels = eps_1 * labels + (1 - eps_1) * labels_g
        grad_get = tf.gradients(self.d_net(input_with_random, labels=labels, is_reuse=True, is_train=True)[0],
                                [input_with_random])[0]
        grad_normal = tf.sqrt(tf.reduce_mean((grad_get) ** 2, axis=1))  # 梯度的二阶范数
        gp_item = 10 * tf.reduce_mean(tf.nn.relu(grad_normal - 1.))  # 对这个限制惩罚项也做了限制
        return gp_item

    # ------------------------------------------------end>>


if __name__ == '__main__':
    print("你不应该在函数库文件中做运行，请找到对应主文件，谢谢")
