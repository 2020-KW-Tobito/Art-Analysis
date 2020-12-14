from os import listdir
from os.path import isfile, join
import codecs


def replaceInFile(file_path, old, newstr):
    f = codecs.open(file_path, 'r', encoding='utf8')
    read_file = f.read()
    f.close()

    new_file = codecs.open(file_path, 'w', encoding='utf8')
    for line in read_file.split("\n"):
        new_file.write(line.replace(old, newstr))
        new_file.write("\n")
    new_file.close()


mypath = "Sketch_svg/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for fname in onlyfiles:
    fpath = join(mypath, fname)
    print(fpath)
    old = '<svg preserveaspectratio="xMinYMin meet" version="1.1" viewbox="0 0 800 800" xmlns="http://www.w3.org/2000/svg">'
    new = '<svg viewBox="0 0 800 800" preserveAspectRatio="xMinYMin meet" xmlns="http://www.w3.org/2000/svg" version="1.1">'
    replaceInFile(fpath, old, new)
