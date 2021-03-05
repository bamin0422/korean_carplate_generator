import os.path     
path = "./DB/lastDB/"

for i in range(0, 60000):
    realPath = path+str(i)+".jpg"
    
    if not os.path.isfile(realPath):
        print(str(i)+".jpg"+" is not in lastDB.")
