import os

dir ="/home/arif/Documents/design/sandpit/sdv2/stablediffusion/outputs/txt2img-samples/samples"
owldir='/home/arif/Documents/design/sandpit/images/owl'
beardir='/home/arif/Documents/design/sandpit/images/bear'
donkeydir='/home/arif/Documents/design/sandpit/images/donkey'
forestdir='/home/arif/Documents/design/sandpit/images/forest'
pigletdir='/home/arif/Documents/design/sandpit/images/piglet'
rabbitdir='/home/arif/Documents/design/sandpit/images/rabbit'
outdirs=[beardir,donkeydir,owldir,pigletdir,rabbitdir,forestdir]

directory = os.fsencode(dir)
imgnum =376
numimages =299
def moveimg(imgnum,dir,outdir):
    imgfile = '/00' + str(imgnum) + '.png' 
    os.replace(dir + imgfile, outdir + imgfile )


for image in range(numimages):
    moveimg(imgnum+image,dir,outdirs[image%6])


