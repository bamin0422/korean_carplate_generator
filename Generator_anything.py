import os, random
import cv2
import numpy as np
import argparse

class ImageGenerator:
    def __init__(self, save_path):
        self.save_path = save_path

    def anything(self, num, save=False):

        cnt=24000
        for i, Iter in enumerate(range(num)):
            imgName = "000"+str(random.randint(0, 2))+str(random.randint(0, 9))+str(random.randint(0, 9))+str(random.randint(0, 9))+str(random.randint(0, 9))
            anything=cv2.imread("background/"+imgName + ".jpg")
            anything = cv2.resize(anything, (260, 55))
            if save:
                cv2.imwrite(self.save_path + "lastDB/" + str(cnt) + ".jpg", anything)
                print("Generate opossite of car plate : "+self.save_path + "lastDB/" + str(cnt) + ".jpg")
            else:
                cv2.imshow(label, anything)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            cnt += 1

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_dir", help="save image directory",
                    type=str, default="./DB/")
parser.add_argument("-n", "--num", help="number of image",
                    type=int, default=6000)
parser.add_argument("-s", "--save", help="save or imshow",
                    type=bool, default=True)
args = parser.parse_args()


img_dir = args.img_dir
imgGenerator = ImageGenerator(img_dir)

num_img = args.num
Save = args.save

imgGenerator.anything(num_img, save=Save)
print("\n Anything finish")
