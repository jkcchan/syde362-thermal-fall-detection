import math
from PIL import Image
"""
interprets bin file and saves a PNG in target directory
"""
def interpret_bin_file_as_img(filename, target):
    file = open(filename)
    hex_list = []
    for line in file:
        hex_list.extend((''.join( [ "%02X " % ord( x ) for x in line ] ).strip()).split(" "))
    new_hex_list = []
    for i, v in enumerate(hex_list):
        if i%2==0:
            new_hex_list.append(hex_list[i]+hex_list[i+1])
    pixels = []
    for i in new_hex_list:
        x = int(i,16)
        pixels.append(int(math.floor(x/256)))
    data = ""
    for i in range(2,60*80+2):
        data += chr(pixels[i]) + chr(pixels[i]) + chr(pixels[i])
    im = Image.frombytes("RGB", (80,60), data)
    im.save(target, "PNG")
"""
interprets bin file and saves a txt in target directory
"""
def interpret_bin_file_as_txt_file(filename, target):
    file = open(filename)
    hex_list = []
    for line in file:
        hex_list.extend((''.join( [ "%02X " % ord( x ) for x in line ] ).strip()).split(" "))
    new_hex_list = []
    for i, v in enumerate(hex_list):
        if i%2==0:
            new_hex_list.append(int(hex_list[i]+hex_list[i+1], 16))
    txt_file = ""
    for i in range(0,60):
        for j in range(0, 80):
            txt_file+=str(new_hex_list[i*j+2])+" "
        txt_file+="\n "
    text_file = open(target, "w")
    text_file.write(txt_file)
    text_file.close()