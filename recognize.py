"""Read an image or directory with the improved recognizer."""
from pathlib import Path
import argparse,json,sys
import cv2,numpy as np
from recognition_pipeline import Recognizer

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,default=Path('recognition_results.jsonl'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--config',type=Path)
    args=p.parse_args()
    files=[args.input] if args.input.is_file() else sorted(f for f in args.input.iterdir() if f.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'})
    if not files:p.error('No images found')
    recognizer=Recognizer(args.config,args.device)
    records=[]
    for start in range(0,len(files),32):
        batch=files[start:start+32]
        images=[cv2.imdecode(np.fromfile(f,dtype=np.uint8),cv2.IMREAD_COLOR) for f in batch]
        if any(im is None for im in images):raise ValueError('Unreadable image')
        preds=recognizer.predict(images)
        for path,(text,score) in zip(batch,preds):
            records.append({'file':str(path),'text':text,'confidence':score})
            print(json.dumps(records[-1],ensure_ascii=False),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(json.dumps(rec,ensure_ascii=False) for rec in records)+'\n',encoding='utf-8')

if __name__=='__main__':main()
