from bata01 import classes_star_gan as m_classes
from bata01 import functions as m_fc
import numpy as np

BACTH_SIZE_TRAIN = 56
BATCH_SIZE_TEST = 40
PATH = "STAR_01"


def main():
    # 首先创建一套用于测试的数据
    random_samples_test = np.random.uniform(low=0., high=1., size=[BATCH_SIZE_TEST, 344])
    random_labels_test = m_fc.read_csv("testdata (2).csv")
    random_labels_test = random_labels_test[0:BATCH_SIZE_TEST]

    A1 = m_classes.CGAN("M_", batch_size=BACTH_SIZE_TRAIN, path=PATH)
    is_reload, rate, new_speed, trained_time_old, now_imgs = m_fc.load_rate("rate-setting.txt")
    trained_times = trained_time_old
    dorl = False
    for i in range(100000000):
        is_reload, new_rate, new_speed, trained_time_old, now_imgs = m_fc.load_rate("rate-setting.txt")
        random_samples = np.random.uniform(low=-1., high=1., size=[BACTH_SIZE_TRAIN, 344])
        random_labels_g = np.random.randint(0, 2, [BACTH_SIZE_TRAIN, 40])*2-1
        random_labels_g = random_labels_g.astype("float")
        data = m_fc.load_img_index("list_attr_celeba.txt")
        sample = m_fc.get_sample(data, BACTH_SIZE_TRAIN)
        images = m_fc.get_image("celeba", sample[:, 0])
        images_in = np.stack(images)
        images_in_test = images_in[0:BATCH_SIZE_TEST]
        labels = sample[:, 1:]
        labels = labels.astype("float")
        labels_test = labels[0:BATCH_SIZE_TEST]
        if trained_times > 0:
           A1.train_net(images_input=images_in, labels=labels, rate=rate, random_samples=random_samples,
                        labels_g=random_labels_g, times=trained_times)
        print(trained_times, rate, dorl)
        if trained_times % 10 == 0:
            d_loss_now, g_loss_now,  g_images_now = A1.test_net(images_input=images_in_test,
                                                                                         labels=labels_test,
                                                                                         random_sample=random_samples_test,
                                                                                         times=trained_times,
                                                                                         labels_g=random_labels_test)
            m_fc.save_img(g_images_now, trained_times, path=PATH)
            now_imgs + 1
            print("dl_loss", d_loss_now, "g_loss",g_loss_now)
        trained_times = trained_times + 1
        if trained_times % 10 == 0:
            rate = rate * new_speed
            if is_reload == 0:
                rate = new_rate * new_speed
            if trained_times%100 == 0:
                m_fc.save_rate(is_reload, rate, new_speed, trained_times, now_imgs, "rate-setting.txt")


if __name__ == "__main__":
    main()
