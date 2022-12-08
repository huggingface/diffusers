import os
from pathlib import Path
from directories import *
import shutil

outdirs=[beardir,donkeydir,owldir,pigletdir,rabbitdir,forestdir,Kangaroodir]

outdirs=[hadacoddir,extoferdir,eyorddir,gwaladir,rawdibdir,rigleldir,winzrigdir,wunzagdir]
outdir_names=['hadacod','extofer','eyord','gwala','rawdib','riglel','winzrig','wunzag']
#outdirs=[womandir]
#outdir_names=['woman']
imgdirs=[beardir,donkeydir,owldir,pigletdir,rabbitdir,forestdir,Kangaroodir,girldir,boydir,womandir]
directory = os.fsencode(dir)



imgnum =0
numimages =379
def moveimg(imgnum,dir,outdir):
    numzeros = 5-len(str(imgnum))
    prefix = '0'*numzeros
    imgfile = '/' + prefix + str(imgnum) +  '.png' 
    #outfile = '/' + prefix + str(imgnum) + '_2'+  '.png' 
    os.replace(dir + imgfile, outdir + imgfile )


def iterateimages():
    for image in range(numimages):
        moveimg(imgnum+image,dir,outdirs[image%len(outdirs)])


def renamefiles(imgdir,prefix):
    count=0
    images = list(Path(imgdir).iterdir())
    for image in images:
        suffix = image.suffix
        newname = prefix + str(count) + suffix
        os.replace(imgdir + '/' + image.name, imgdir + '/' + newname )
        count +=1



def move_all_imgs(fromdir, todir):
    prompt_images = list(Path(fromdir).iterdir())
    for image in prompt_images:
        shutil.copyfile(fromdir + '/'+ image.name, todir +'/'+ image.name )

def move_images():
    for fromdir in imgdirs:
        move_all_imgs(fromdir,all_imges)


def rename_images():
    for idx,dir in enumerate(outdirs):
        renamefiles(dir, outdir_names[idx])


rename_images()