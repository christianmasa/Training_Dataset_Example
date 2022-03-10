import os
import json
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
import  numpy as np
import aiofiles
import asyncio
from threading import Thread


def write_file(line, dest):
    with open(dest, 'w+') as file:
        file.write(('%g ' * len(line)).rstrip() % line + '\n')
        
def write_file_list(filenames, dest):
    with open(dest, 'w+') as file:
        for line in filenames:
            file.write('%s' % line + '\n')
        
def convert_coco_json(json_dir='../coco/annotations/',output_dir='../coco/labels/',
       image_dir="../coco/data", text_dir='../coco/', batch_size=256, use_segments=False):
    if json_dir == None or output_dir ==None or text_dir==None or image_dir==None:
        print("Directories can't be none")
        return
    os.makedirs(output_dir,exist_ok=True)
    jsons = glob.glob(json_dir + '*.json')

    fn = Path(output_dir)
    
    # Import json
    for json_file in sorted(jsons):
        sub_type = 'train' if 'train' in json_file else 'val'
        with open(json_file,'r') as f:
            data = json.load(f)

        # Create image dict
        images = {'%g' % x['id']: x for x in data['images']}

        filenames = []
        
        tasks = []
        i = 0
        # Write labels file
        for x in tqdm(data['annotations'], desc='Annotations %s' % json_file,  position=0, leave=True):
            if x['iscrowd']:
                continue

            img = images['%g' % x['image_id']]
            h, w, f = img['height'], img['width'], img['file_name'].split("/")[-1]
            filenames.append(os.path.join(image_dir,f))
            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(x['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            # Segments
            s = None
            if use_segments:
                segments = [j for i in x['segmentation'] for j in i]  # all segments concatenated
                s = (np.array(segments).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()

            # Write
            if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
                line = x['category_id'], *(s if use_segments else box)  # cls, box or segments
                
                #th = Thread(target=write_file, args=(line,(fn / f).with_suffix('.txt')))
                task = Thread(target=write_file,args = (line,(fn / f).with_suffix('.txt')))
                task.start()
                tasks.append(task)
            
            if i == batch_size-1:
                [t.join() for t in tasks]
                tasks = []
            
            i = (i + 1 ) % batch_size
        filenames = list(set(filenames))          
        task = Thread(target=write_file_list, args=(filenames, os.path.join(text_dir, sub_type+".txt")))
        task.start()
        tasks.append(task)
        if tasks:
            [t.join() for t in tasks]
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-dir', type=str, default=None, help='jsons folder path')
    parser.add_argument('--output-dir', type=str, default=None, help='labels output folder path')
    parser.add_argument('--image-dir', type=str, default=None, help='labels output folder path')
    parser.add_argument('--text-dir', type=str, default=None, help='labels output folder path')
    args = parser.parse_args()
    
    convert_coco_json(args.json_dir, args.output_dir, args.image_dir,args.text_dir)
    