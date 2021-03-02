import os, random
import cv2
import numpy as np
import argparse

def image_augmentation(img, ang_range=6, shear_range=3, trans_range=3):
    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 0.9)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, Rot_M, (cols, rows))
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    img = cv2.warpAffine(img, shear_M, (cols, rows))

    # Brightness
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .4 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # Blur
    blur_value = random.randint(0,3) * 2 + 1
    img = cv2.blur(img,(blur_value, blur_value))

    return img


class ImageGenerator:
    def __init__(self, save_path):
        self.save_path = save_path
        # Plate
        self.plate = cv2.imread("plate.jpg")
        self.markPlate = cv2.imread("mark_plate.png")

        # loading Number ====================  white-one-line  ==========================
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

    def ver2006(self, num, save=False):
        number = [cv2.resize(number, (25, 38)) for number in self.Number]
        char = [cv2.resize(char1, (27, 38)) for char1 in self.Char1]
        cnt = 18000

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (234, 50))
            random_width = 55
            random_height = 260
            random_R, random_G, random_B = random.randint(0,255), random.randint(0,255), random.randint(0,255)
            background = np.zeros((random_width, random_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (random_width, random_height), (random_R, random_G, random_B), -1)

            # row -> y , col -> x
            row, col = 7, 18  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            Plate[row:row + 38, col:col + 25, :] = number[rand_int]
            col += 25

            # number 2
            rand_int = random.randint(0, 9)
            Plate[row:row + 38, col:col + 25, :] = number[rand_int]
            col += 25

            # character 3
            Plate[row:row + 38, col:col + 27, :] = char[i%37]
            col += (27 + 18)

            # number 4
            rand_int = random.randint(0, 9)
            Plate[row:row + 38, col:col + 25, :] = number[rand_int]
            col += 25

            # number 5
            rand_int = random.randint(0, 9)
            Plate[row:row + 38, col:col + 25, :] = number[rand_int]
            col += 25

            # number 6
            rand_int = random.randint(0, 9)
            Plate[row:row + 38, col:col + 25, :] = number[rand_int]
            col += 25 

            # number 7
            rand_int = random.randint(0, 9)
            Plate[row:row + 38, col:col + 25, :] = number[rand_int]
            col += 25

            random_w = random.randint(0, random_width - 55)
            random_h = random.randint(0, random_height - 260)
            background[random_w:50+random_w, random_h:234+random_h, :] = Plate
            background = image_augmentation(background)

            if save:
                cv2.imwrite(self.save_path + "lastDB/" + str(cnt) + ".jpg", background)
                print("Generate augmentated car plate 2006 : "+self.save_path + "lastDB/" + str(cnt) + ".jpg")
                cnt += 1
            else:
                cv2.imshow(cnt, background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def ver2019(self, num, save=False):
        number = [cv2.resize(number, (23, 34)) for number in self.Number]
        char = [cv2.resize(char1, (24, 34)) for char1 in self.Char1]
        cnt = 20000

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (234, 50))
            random_width = 55
            random_height = 260
            random_R, random_G, random_B = random.randint(0,255), random.randint(0,255), random.randint(0,255)
            background = np.zeros((random_width, random_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (random_width, random_height), (random_R, random_G, random_B), -1)

            # row -> y , col -> x
            row, col = 7, 18  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            Plate[row:row + 34, col:col + 23, :] = number[rand_int]
            col += 23

            # number 2
            rand_int = random.randint(0, 9)
            Plate[row:row + 34, col:col + 23, :] = number[rand_int]
            col += 23

            # number 3
            rand_int = random.randint(0, 9)
            Plate[row:row + 34, col:col + 23, :] = number[rand_int]
            col += 23

            # hangeul
            Plate[row:row + 34, col:col + 24, :] = char[i%37]
            col += (20 + 16)

            # number 4
            rand_int = random.randint(0, 9)
            Plate[row:row + 34, col:col + 23, :] = number[rand_int]
            col += 23

            # number 5
            rand_int = random.randint(0, 9)
            Plate[row:row + 34, col:col + 23, :] = number[rand_int]
            col += 24

            # number 6
            rand_int = random.randint(0, 9)
            Plate[row:row + 34, col:col + 23, :] = number[rand_int]
            col += 23

            # number 7
            rand_int = random.randint(0, 9)
            Plate[row:row + 34, col:col + 23, :] = number[rand_int]
            col += 23

            random_w = random.randint(0, random_width - 55)
            random_h = random.randint(0, random_height - 260)
            background[random_w:50+random_w, random_h:234+random_h, :] = Plate
            background = image_augmentation(background)

            if save:
                cv2.imwrite(self.save_path + "lastDB/" + str(cnt) + ".jpg", background)
                print("Generate augmentated carplate 2019 : "+self.save_path + "lastDB/" + str(cnt) + ".jpg")
                cnt += 1
            else:
                cv2.imshow(cnt, background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def mark_ver2019(self, num, save=False):
        number = [cv2.resize(number, (23, 34)) for number in self.Number]
        char = [cv2.resize(char1, (24, 34)) for char1 in self.Char1]
        cnt = 22000

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.markPlate, (234, 50))
            random_width = 55
            random_height = 260
            random_R, random_G, random_B = random.randint(0,255), random.randint(0,255), random.randint(0,255)
            background = np.zeros((random_width, random_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (random_width, random_height), (random_R, random_G, random_B), -1)

            # row -> y , col -> x
            row, col = 7, 33  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            Plate[row:row + 34, col:col + 23, :] = number[rand_int]
            col += 23

            # number 2
            rand_int = random.randint(0, 9)
            Plate[row:row + 34, col:col + 23, :] = number[rand_int]
            col += 23

            # number 3
            rand_int = random.randint(0, 9)
            Plate[row:row + 34, col:col + 23, :] = number[rand_int]
            col += 23

            # hangeul
            Plate[row:row + 34, col:col + 24, :] = char[i%37]
            col += (20 + 16)

            # number 4
            rand_int = random.randint(0, 9)
            Plate[row:row + 34, col:col + 23, :] = number[rand_int]
            col += 23

            # number 5
            rand_int = random.randint(0, 9)
            Plate[row:row + 34, col:col + 23, :] = number[rand_int]
            col += 24

            # number 6
            rand_int = random.randint(0, 9)
            Plate[row:row + 34, col:col + 23, :] = number[rand_int]
            col += 23

            # number 7
            rand_int = random.randint(0, 9)
            Plate[row:row + 34, col:col + 23, :] = number[rand_int]
            col += 23

            random_w = random.randint(0, random_width - 55)
            random_h = random.randint(0, random_height - 260)
            background[random_w:50+random_w, random_h:234+random_h, :] = Plate
            background = image_augmentation(background)

            if save:
                cv2.imwrite(self.save_path + "lastDB/" + str(cnt) + ".jpg", background)
                print("Generate augmentated carplate mark_2019 : "+self.save_path + "lastDB/" + str(cnt) + ".jpg")
                cnt += 1
            else:
                cv2.imshow(cnt, background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_dir", help="save image directory",
                    type=str, default="./DB/")
parser.add_argument("-n", "--num", help="number of image",
                    type=int, default=2000)
parser.add_argument("-s", "--save", help="save or imshow",
                    type=bool, default=True)
args = parser.parse_args()


img_dir = args.img_dir
imgGenerator = ImageGenerator(img_dir)

num_img = args.num
Save = args.save

imgGenerator.ver2006(num_img, save=Save)
print("ver2006 finish")
imgGenerator.ver2019(num_img, save=Save)
print("ver2019 finish")
imgGenerator.mark_ver2019(num_img, save=Save)
print("mark_ver2019 finish")