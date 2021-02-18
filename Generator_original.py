import os, random
import cv2, argparse
import time
import numpy as np

def random_bright(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

class ImageGenerator:
    def __init__(self, save_path):
        self.save_path = save_path
        # Plate
        self.plate = cv2.imread("plate.jpg")
        self.plate2 = cv2.imread("plate_y.jpg")
        self.plate3 = cv2.imread("plate_g.jpg")

        # loading Number
        file_path = "./num/"
        file_list = os.listdir(file_path)
        self.Number = list()
        self.number_list = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Number.append(img)
            self.number_list.append(file[0:-4])

        # loading Char
        file_path = "./char1/"
        file_list = os.listdir(file_path)
        self.char_list = list()
        self.Char1 = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Char1.append(img)
            self.char_list.append(file[0:-4])

    def Type_1(self, num, save=False):
        number = [cv2.resize(number, (28, 42)) for number in self.Number]
        char = [cv2.resize(char1, (30, 42)) for char1 in self.Char1]
        Plate = cv2.resize(self.plate, (260, 55))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (260, 55))
            label = "Z"
            # row -> y , col -> x
            row, col = 7, 18  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 42, col:col + 28, :] = number[rand_int]
            col += 28

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 42, col:col + 28, :] = number[rand_int]
            col += 28

            # character 3
            label += self.char_list[i%37]
            Plate[row:row + 42, col:col + 30, :] = char[i%37]
            col += (30 + 18)

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 42, col:col + 28, :] = number[rand_int]
            col += 28

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 42, col:col + 28, :] = number[rand_int]
            col += 28

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 42, col:col + 28, :] = number[rand_int]
            col += 28

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 42, col:col + 28, :] = number[rand_int]
            col += 28
            Plate = random_bright(Plate)
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                tiem.sleep(10)
                cv2.destroyAllWindows()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_dir", help="save image directory",
                    type=str, default="./DB/")
parser.add_argument("-n", "--num", help="number of image",
                    type=int, default=10)
parser.add_argument("-s", "--save", help="save or imshow",
                    type=bool, default=False)
args = parser.parse_args()


img_dir = args.img_dir
imgGenerator = ImageGenerator(img_dir)

num_img = args.num
print(args)
Save = args.save

imgGenerator.Type_1(num_img, save=Save)
print("Type 1 finish")
