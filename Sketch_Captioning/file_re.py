import os

datadir = "./hymenoptera_data/data/sketches_png/"

files = os.listdir(datadir)
filename = []

for i in files:
    filename.append(i)

print(filename)