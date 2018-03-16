def interpret_bin_file(filename, target):
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