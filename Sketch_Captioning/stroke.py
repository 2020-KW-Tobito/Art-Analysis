from IPython.display import SVG
from bs4 import BeautifulSoup
import os
from pandas import DataFrame
from pandas import ExcelFile

import numpy

datadir = "./Sketch_svg"

files = os.listdir(datadir)
categories=[]

for i in files:
    categories.append(i)

print(categories)
num_classes=len(categories)

map_svg = None
directory = "Sketch_svg/"

for c in range(240):
    file = categories[c]
    svg_file = directory + file

    with open(svg_file, 'r') as f:
        map_svg = f.read()

    soup = BeautifulSoup(map_svg)
    # paths = soup.select('path[id]')

##print(paths)
    #
    # colors = ['#6e36c9', '#e61710', '#f4f716', '#DF65B9', '#2ae615', '#2547cf']
    #
    # i=0
    #
    # for p in paths:
    #     i+=1
    #     if(i>5):
    #         i=0
    #     p['stroke'] = colors[i]
    #     print(p)
    # new_svg = soup.prettify()
    # SVG(new_svg)
    #
    # new_svg.__sizeof__()
    #
    # print(new_svg)
    #
    # file_svg= "Sketch_svg/"
    # files_name = file_svg + file
    # with open(files_name, 'w') as f:
    #     f.write(new_svg)

    from cairosvg import svg2png
    s=os.path.splitext(svg_file)
    s= os.path.split(s[0])
    file_png = ".png"
    files_ = s[1]+file_png
    files_ = "Sketch_line/" + files_
    svg2png(url=svg_file, write_to=(files_), output_height=1300, output_width=1300)