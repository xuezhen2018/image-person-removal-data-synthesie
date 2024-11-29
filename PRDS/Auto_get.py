import sys
import os
import numpy as np
import time

from mlagents.envs.environment import UnityEnvironment
from PIL import Image

SYN_DATA_DIR = "E:/Xue/Unity/pic"


class GetPic():
    def __init__(self):
        self.epoch =15

    def get_unity_envs(self):
        # check the python environment
        print("Python version: ", sys.version)
        if (sys.version_info[0] < 3):
            raise Exception("ERROR: ML-Agents Toolkit requires Python 3")

        # set the unity environment
        env = UnityEnvironment(file_name=None, base_port=5005)

        # set the default brain to work with
        brain = env.brain_names[0]

        return env, brain
    def get_images_by_attributes(self,env, brain):

        # print(self.num)
        print("start get images")
        # env.reset(train_mode=True)[brain]
        P_path = os.path.join(SYN_DATA_DIR, "image")

        R_path = os.path.join(SYN_DATA_DIR, "R_image")
        M_path = os.path.join(SYN_DATA_DIR, "m_image")

        if not os.path.exists(P_path):
            os.makedirs(P_path)
        if not os.path.exists(R_path):
            os.makedirs(R_path)
        if not os.path.exists(M_path):
            os.makedirs(M_path)
        # while True:

        self.father = 0
        for j in range(35):
            self.person_n = 1
            for name in range(10):
                self.num = self.epoch * (self.person_n-1) + self.father * 150
                print(self.num)
                i =0
                self.save_count = 0
                while i != 20:
                    self.num += 1

                    print("env_info")
                    env_info = env.step([self.save_count,name, self.father, self.father + 1, self.person_n])[brain]
                    print("start save photo")
                    if self.save_count == 15:
                        break
                    p_img = np.asarray(env_info.visual_observations[0][0] * 255.0, dtype=np.uint8)
                    b_img = np.asarray(env_info.visual_observations[1][0] * 255.0, dtype=np.uint8)
                    w_img = np.asarray(env_info.visual_observations[2][0] * 255.0, dtype=np.uint8)
                    r_img = np.asarray(env_info.visual_observations[3][0] * 255.0, dtype=np.uint8)
                    if b_img.any() != 0:
                        p_img_s = Image.fromarray(p_img)
                        p_img_s.save(P_path + "/%d.png" % (self.num))
                        b_img_s = Image.fromarray(b_img)
                        # b_img_s.save(B_path + "/%d.png" % (self.num))
                        w_img_s = Image.fromarray(w_img)
                        # w_img_s.save(W_path + "/%d.png"%(self.num))
                        r_img_s = Image.fromarray(r_img)
                        r_img_s.save(R_path + "/%d.png"%(self.num))

                        #get mask
                        #
                        rest = (b_img == 0) & (w_img == 255)
                        mask = (rest == False)
                        img_copy = b_img.copy()
                        img_copy[mask] = 255
                        m_img = Image.fromarray(img_copy)
                        m_img.save(M_path + "/%d.png"%(self.num))
                        print("save No.",self.num,"photo·")
                        self.save_count = self.save_count + 1
                        i = i + 1
                    else:
                        self.num -= 1
                        i = i-1

                self.person_n = self.person_n +1
            self.father = self.father +1
if __name__ == "__main__":
    # get unity environment
    start = time.time()
    GP = GetPic()
    print("get env，brain")
    env, brain = GP.get_unity_envs()
    print("got it")
    GP.get_images_by_attributes(env,brain)
    end = time.time()
    SPtime = end-start
    print(SPtime)
    env.close()