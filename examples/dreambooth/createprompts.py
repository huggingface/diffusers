import json
from pathlib import Path
import pickle
from directories import *


imgdirs=[extoferdir,eyorddir,gwaladir,hadacoddir,rawdibdir,renwadir,rigleldir,ungeradir,wunzagdir]
imgprompts=["A cartoon image of extofer","A cartoon image of an eyord donkey","A cartoon image of a gwala owl","A cartoon image of a hadacod forest","A cartoon image of a rawdib rabbit","An image of renwa","A cartoon image of a riglel piglet","A cartoon image of an ungera kangaroo","A cartoon image of a wunzag bear"]
promptpicklename='instanceprompts.pickle'
#imgdirs=[beardir,boydir,donkeydir,forestdir,girldir,womandir,Kangaroodir,owldir,pigletdir,rabbitdir]
#imgprompts=["A cartoon image of a bear","A cartoon image of a boy","A cartoon image of a donkey","A cartoon image of a forest","A cartoon image of a girl","An image of a woman","A cartoon image of a kangaroo","A cartoon image of an owl","A cartoon image of a piglet","A cartoon image of a rabbit"]
#promptpicklename='classprompts.pickle'
def get_prompts(promptsdir, filename):
    with open(promptsdir + '/' +filename , 'rb') as handle:
        b = pickle.load(handle)
    return b

def create_promptfile(imageprompt,imgdir,promptdir,filename,loadfromdir=False):
    if loadfromdir:
        imageprompts= get_prompts(promptdir,filename)
    else:
        imageprompts={}
    prompt_images = list(Path(imgdir).iterdir())
    for image in prompt_images:
        imageprompts[image.name]=imageprompt


    with open(promptsdir + '/'+filename, 'wb') as handle:
        pickle.dump(imageprompts, handle, protocol=pickle.HIGHEST_PROTOCOL)




#create_promptfile(dir,promptsdir)

def create_prompts():
    create_promptfile(imgprompts[0],imgdirs[0],promptsdir,promptpicklename,False)
    for idx in range(1,len(imgdirs)):
        create_promptfile(imgprompts[idx],imgdirs[idx],promptsdir,promptpicklename,True)

get_prompts(promptsdir,promptpicklename)
#create_prompts()



def find_missing_images():
    dir ='/home/arif/Documents/design/sandpit/allimages'
    prompt_images = list(Path(dir).iterdir())
    prompts=get_prompts('/home/arif/Documents/design/sandpit/prompts','classprompts.pickle')
    for image in prompt_images:
        res=prompts.get(image.name)
        if res==None:
            print('not founc')

