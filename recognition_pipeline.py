"""Versioned inference using a frozen development-selected configuration."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,cv2,numpy as np
import recognition_model as r
from rapidfuzz.distance import Levenshtein
from recognition_transforms import crop_center,widen_gap,adaptive_crop,clipped_ink_pixels
from rectifier import rectify_image

def choose(candidates,policy):
    method=policy.get('method','confidence')
    word_bonus=policy.get('word_bonus',0.0)
    def quality(c):return c[1]+word_bonus*(len(c[0].split())==2)
    if method=='confidence':return max(candidates,key=quality)
    if method=='vote':
        totals=defaultdict(float)
        for text,score in candidates:totals[text]+=max(.01,score)
        best=max(totals,key=lambda t:(totals[t],max(quality(c) for c in candidates if c[0]==t)))
        return max((c for c in candidates if c[0]==best),key=quality)
    if method=='longest_close':
        floor=max(quality(c) for c in candidates)-policy.get('tolerance',.03)
        return max((c for c in candidates if quality(c)>=floor),key=lambda c:(len(c[0].replace(' ','')),quality(c)))
    raise ValueError('Unknown candidate selection method')

class Recognizer:
    def __init__(self,config_path=None,device='cpu'):
        self.config_path=Path(config_path or r.ROOT/'experiments/recognition_v2/frozen_config.json')
        self.config=json.loads(self.config_path.read_text(encoding='utf-8'))
        weights=(self.config_path.parent/self.config['weights']).resolve()
        actual_hash=hashlib.sha256(weights.read_bytes()).hexdigest()
        if actual_hash!=self.config['weights_sha256']:
            raise ValueError('Checkpoint differs from the validated frozen configuration.')
        self.model=r.build(weights,device)
        self.rescue_model=None
        if self.config.get('insertion_rescue'):
            info=self.config['insertion_rescue']
            source=(self.config_path.parent/info['weights']).resolve()
            if hashlib.sha256(source.read_bytes()).hexdigest()!=info['weights_sha256']:
                raise ValueError('Rescue checkpoint hash mismatch.')
            self.rescue_model=r.build(source,device)
    def predict_rectified(self,images):
        all_predictions=[]
        for v in self.config['variants']:
            prepared=[]
            for image in images:
                if v.get('widen_gap',False):image,_=widen_gap(image,extra=v.get('gap_extra',12))
                if v.get('adaptive'):
                    image=adaptive_crop(image,v.get('minimum',40),v.get('margin',4))
                    if v.get('adaptive_normalize'):
                        image=cv2.resize(image,(image.shape[1],40))
                else:
                    image=crop_center(image,v['crop'],v.get('vertical_offset',0))
                prepared.append(image)
            all_predictions.append(r.predict(self.model,prepared,v['constrained'],v['scale'],
                                              beam_width=v.get('beam_width',0)))
        policy=self.config['selection']
        if policy['method']=='boundary_guard':
            result=[]
            for i,image in enumerate(images):
                split=policy['base_count']
                base=choose([p[i] for p in all_predictions[:split]],policy['base_policy'])
                guard=choose([p[i] for p in all_predictions[split:]],policy['guard_policy'])
                base_length=len(base[0].replace(' ',''));guard_length=len(guard[0].replace(' ',''))
                length_mode=policy.get('length_mode','any')
                eligible=(length_mode=='any' or (guard_length>base_length if length_mode=='longer' else guard_length>=base_length))
                result.append(guard if eligible and clipped_ink_pixels(image)>policy['threshold'] else base)
        else:
            result=[choose([pred[i] for pred in all_predictions],policy) for i in range(len(images))]
        if self.rescue_model is not None:
            info=self.config['insertion_rescue']
            indices=[i for i,im in enumerate(images) if clipped_ink_pixels(im)>info['threshold']]
            old=r.predict(self.rescue_model,[images[i] for i in indices],False,1.0)
            for i,(text,score) in zip(indices,old):
                original=result[i][0]
                a=original.replace(' ','');b=text.replace(' ','')
                if (score>=info['minimum_confidence'] and len(b)==len(a)+1 and
                    Levenshtein.distance(a,b)==1 and len(text.split())==len(original.split()) and
                    all(ch==' ' or 'а'<=ch<='я' or ch=='ё' for ch in text)):
                    result[i]=(text,score)
        return result
    def predict(self,images):
        rectified=[]
        for image in images:
            try:
                rectified.append(rectify_image(image)[0])
            except (RuntimeError,ValueError,cv2.error,np.linalg.LinAlgError):
                rectified.append(image)
        return self.predict_rectified(rectified)
