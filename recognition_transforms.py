"""Image-only spacing transforms; no dictionaries or reference answers."""
import cv2,numpy as np

def crop_center(image,height,offset=0):
    height=min(height,image.shape[0])
    top=max(0,min(image.shape[0]-height,(image.shape[0]-height)//2+offset))
    return image[top:top+height]

def adaptive_crop(image,minimum=40,margin=4):
    """Expand the line crop when text components extend past fixed margins."""
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    bh=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
    threshold,mask=cv2.threshold(bh,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask=(bh>max(threshold,24)).astype('uint8')
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    boxes=[]
    for (x,y,w,h,area),(cx,cy) in zip(stats[1:],centers[1:]):
        if 2<=w<=45 and 5<=h<=40 and 10<=area<=700:
            boxes.append((y,y+h))
    if len(boxes)<4:return crop_center(image,minimum)
    lo=max(0,min(b[0] for b in boxes)-margin)
    hi=min(image.shape[0],max(b[1] for b in boxes)+margin)
    if hi-lo<minimum:
        middle=(lo+hi)//2
        lo=max(0,min(image.shape[0]-minimum,middle-minimum//2))
        hi=lo+minimum
    return image[lo:hi]

def clipped_ink_pixels(image,top=8,bottom=52):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    bh=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
    threshold,_=cv2.threshold(bh,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask=(bh>max(threshold,24)).astype('uint8')
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    retained=np.zeros_like(mask)
    for i,(x,y,w,h,area) in enumerate(stats[1:],1):
        if 2<=w<=45 and 5<=h<=40 and 10<=area<=700:
            retained[labels==i]=1
    return int(retained[:top].sum()+retained[bottom:].sum())

def widen_gap(image,extra=12):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    bh=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
    threshold,_=cv2.threshold(bh,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask=bh>max(24,threshold)
    # Discard remapping artifacts far outside the line's central band.
    band=crop_center(mask,32)
    occupied=band.sum(axis=0)>=2
    xs=np.flatnonzero(occupied)
    if len(xs)<12:return image,False
    gaps=[]
    for left,right in zip(xs[:-1],xs[1:]):
        size=int(right-left-1)
        center=(left+right)//2
        if size>=3 and .2*image.shape[1]<center<.8*image.shape[1]:
            gaps.append((size,center))
    if not gaps:return image,False
    gaps.sort(reverse=True)
    size,center=gaps[0]
    # Avoid turning a normal inter-letter gap into a fabricated word boundary.
    if size<4 or (len(gaps)>1 and size<1.45*gaps[1][0]):return image,False
    fill=np.repeat(image[:,center:center+1],extra,axis=1)
    return np.concatenate([image[:,:center],fill,image[:,center:]],axis=1),True
