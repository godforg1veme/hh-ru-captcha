"""Shared model loading, preprocessing, and Cyrillic CTC decoding."""
from pathlib import Path
import os,sys,copy,json
import numpy as np
ROOT=Path(__file__).resolve().parent
# Keep CUDA libraries local to this environment, without changing system PATH.
_DLL_HANDLES=[]
for base in [Path(sys.prefix)/'Lib/site-packages/nvidia',Path(sys.prefix)/'Lib/site-packages/paddle/libs']:
    if base.exists():
        for directory in ([base] if base.name=='libs' else list(base.glob('*/bin'))):
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
            os.environ['PATH']=str(directory)+os.pathsep+os.environ.get('PATH','')
sys.path.insert(0,str(ROOT/'PaddleOCR'))
import cv2,paddle,yaml
from ppocr.modeling.architectures import build_model

MODEL_DIR=ROOT/'PaddleOCR/models/eslav_PP-OCRv5_mobile_rec_infer/eslav_PP-OCRv5_mobile_rec_infer'
PRETRAINED=ROOT/'PaddleOCR/models/eslav_PP-OCRv5_mobile_rec_pretrained.pdparams'
CHARACTERS=['blank']+(MODEL_DIR/'ppocr_keys.txt').read_text(encoding='utf-8').splitlines()+[' ']
assert len(CHARACTERS)==519
RUSSIAN=' абвгдеёжзийклмнопрстуфхцчшщъыьэюя'

def build(weights=None,device='cpu'):
    paddle.set_device(device)
    config=yaml.safe_load((ROOT/'PaddleOCR/configs/rec/PP-OCRv5/multi_language/eslav_PP-OCRv5_mobile_rec.yml').read_text(encoding='utf-8'))['Architecture']
    config['Head']['out_channels_list']={'CTCLabelDecode':len(CHARACTERS),'NRTRLabelDecode':len(CHARACTERS)+3}
    model=build_model(config)
    state=paddle.load(str(weights or PRETRAINED))
    own=model.state_dict()
    assert set(state)==set(own), (len(state),len(own),set(own)-set(state))
    assert all(list(own[k].shape)==list(v.shape) for k,v in state.items())
    model.set_state_dict(state)
    # The auxiliary attention head is not used by the deployed CTC recognizer.
    for name,param in model.named_parameters():
        if name.startswith(('head.gtc_head.','head.before_gtc.')):
            param.stop_gradient=True
    model.eval()
    return model

def forward(model,x):
    return model.head.ctc_head(model.head.ctc_encoder(model.backbone(x)))

def preprocess(image,width_scale=1.0,height=48):
    width=min(320,max(8,round(height*image.shape[1]/image.shape[0]*width_scale)))
    resized=cv2.resize(image,(width,height)).astype('float32')/127.5-1
    out=np.zeros((3,height,320),dtype='float32')
    out[:,:,:width]=resized.transpose(2,0,1)
    return out

def character_probabilities(probs,constrained=False):
    probs=np.asarray(probs)
    if constrained:
        # Merge uppercase/lowercase probability mass before CTC collapse.
        # This is a Russian lowercase task; no word lexicon or labels are used.
        chars=['blank']+list(RUSSIAN)
        q=np.zeros((probs.shape[0],len(chars)),dtype='float32')
        q[:,0]=probs[:,0]
        for j,ch in enumerate(CHARACTERS[1:],1):
            lower=ch.casefold()
            if lower in RUSSIAN and len(lower)==1:
                q[:,chars.index(lower)]+=probs[:,j]
        probs=q
    else:
        chars=CHARACTERS
    return probs,chars

def decode(probs,constrained=False):
    probs,chars=character_probabilities(probs,constrained)
    indices=probs.argmax(axis=1)
    selected=(indices!=0)&np.r_[True,indices[1:]!=indices[:-1]]
    text=''.join(chars[i] for i in indices[selected])
    confidence=float(probs[np.where(selected)[0],indices[selected]].mean()) if selected.any() else 0.0
    return ' '.join(text.casefold().split()),confidence

def decode_beam(probs,beam_width=8,top_k=4):
    """Lexicon-free CTC prefix beam search, including blank-separated repeats."""
    probs,chars=character_probabilities(probs,True)
    logp=np.log(np.maximum(probs,1e-12))
    beams={(): (0.0,float('-inf'))}
    add=lambda a,b:float(np.logaddexp(a,b))
    for frame in logp:
        indices=set(np.argsort(frame)[-top_k:].tolist())|{0}
        nxt={}
        def update(prefix,blank=None,nonblank=None):
            pb,pn=nxt.get(prefix,(float('-inf'),float('-inf')))
            nxt[prefix]=(add(pb,blank) if blank is not None else pb,
                         add(pn,nonblank) if nonblank is not None else pn)
        for prefix,(pb,pn) in beams.items():
            total=add(pb,pn)
            update(prefix,blank=total+float(frame[0]))
            for c in indices-{0}:
                value=float(frame[c])
                if prefix and c==prefix[-1]:
                    update(prefix,nonblank=pn+value)
                    update(prefix+(c,),nonblank=pb+value)
                else:
                    update(prefix+(c,),nonblank=total+value)
        beams=dict(sorted(nxt.items(),key=lambda kv:add(*kv[1]),reverse=True)[:beam_width])
    prefix,states=max(beams.items(),key=lambda kv:add(*kv[1]))
    text=' '.join(''.join(chars[i] for i in prefix).split())
    score=float(np.exp(min(0.0,add(*states)/max(1,len(prefix)))))
    return text,score

def predict(model,images,constrained=False,width_scale=1.0,batch_size=16,beam_width=0):
    result=[]
    with paddle.no_grad():
        for start in range(0,len(images),batch_size):
            x=np.stack([preprocess(im,width_scale) for im in images[start:start+batch_size]])
            probs=forward(model,paddle.to_tensor(x)).numpy()
            result.extend((decode_beam(p,beam_width) if beam_width else decode(p,constrained)) for p in probs)
    return result
