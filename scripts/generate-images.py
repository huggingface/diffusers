#!/usr/bin/env python

from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline
import time,re,sys,os,os.path
from datetime import datetime

def main():
    model_id = "google/ddpm-celebahq-256"

    argc = len( sys.argv )

    if 2 <= argc:
        model_id = sys.argv[ 1 ];
        if "-h" == model_id:
            print("usage: generate-images.py [model-id|output-directory] [count]")
            exit(0)

    count = 1
    if 3 <= argc:
        count = int( sys.argv[ 2 ] )

    if os.path.exists(model_id):
        mtime = os.path.getmtime(model_id)
        t = datetime.fromtimestamp(mtime)
        now = t.strftime('%Y-%m-%d+%H-%M-%S')
        print(f"modification time on {model_id} is {now}")
    else:
        t = datetime.now()
        now = t.strftime('%Y-%m-%d+%H-%M-%S')
        print(f"using current time {now}")

    # load model and scheduler
    print( f"loading the model from {model_id}" )
    pipeline = DDPMPipeline.from_pretrained(model_id)  
    # these didn't seem to work for me with my trained data
    #pipeline = PNDMPipeline.from_pretrained(model_id) 
    #pipeline = DDIMPipeline.from_pretrained(model_id) 
    print( f"loaded the model from {model_id}" )

    base = os.path.basename(model_id)
    directory = f"generated/{base}/{now}"
    os.makedirs(directory, exist_ok=True)

    html = '''<!DOCTYPE html PUBLIC"-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<HTML xmlns="http://www.w3.org/1999/xhtml">
    <HEAD>
        <meta content="text/html;charset=utf-8" http-equiv="Content-Type">
        <meta content="utf-8" http-equiv="encoding">

        <TITLE>334</TITLE>
        <script type="text/javascript">
            window.addEventListener('load', () => {} );
        </script>
        <style>
            body { color:#ccb; background:black; font-family: sans-serif; margin:.5em; }
            pre  { color:#8c8; }  
            a    { color:#aad; text-decoration:none; }
        </style>
    </HEAD>
    <BODY>
    '''

    i = -1
    while i < count - 1:
        i = i + 1
        id = str(i).zfill( 4 )
        basename = f"image-{id}.png"
        filename = f"{directory}/{basename}"
        if os.path.exists(filename):
            print(f"skipping {i} because {filename} already exists")
            count = count + 1 
            continue
        # run pipeline in inference (sample random noise and denoise) and save 
        print( f"creating image and saving to {filename}" )
        image = pipeline()["sample"]
        image[0].save(filename)
        print( f"image saved to {filename}" )
        link = f'\t\t<span><a href="{basename}"><img src="{basename}"></img></a></span>\n'
        html = html + link

    html = html + "\t</BODY>\n</HTML>"
    filename = f"{directory}/images.html"
    print(f"writing html to {filename}")
    f = open( filename, "w" )
    f.write( html )
    f.close()
    print(f"wrote html to {filename}")


main()
