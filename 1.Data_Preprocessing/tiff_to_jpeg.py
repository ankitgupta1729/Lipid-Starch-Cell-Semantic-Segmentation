import os, sys
from PIL import Image
def tiff_to_jpeg():
    for infile in os.listdir(os.getcwd()+"/sample"):
        print("file name: " + infile)
        if infile[-4:] == "tiff":
            outfile = infile[:-4] + "jpeg"
            img = Image.open(os.getcwd()+"/sample/"+infile)
            print("new filename : " + outfile)
            out = img.convert("RGB")
            out.save(os.getcwd()+"/out/"+outfile, "JPEG", quality=90)
        if infile[-3:] == "tif":
            outfile = infile[:-3] + "jpeg"
            img = Image.open(os.getcwd()+"/sample/"+infile)
            print("new filename : " + outfile)
            out = img.convert("RGB")
            out.save(os.getcwd()+"/out/"+outfile, "JPEG", quality=90)
            
tiff_to_jpeg()            