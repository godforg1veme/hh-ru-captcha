"""Curved Cyrillic research images. Dependencies: Pillow, NumPy."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parent
FINAL_W, FINAL_H, SCALE = 250, 90, 3
DEFAULT_OUTPUT = ROOT / 'synthetic_v4'
DEFAULT_PHRASES = [
    'амил ботиночною', 'засушенной авне', 'ассамская плюёт', 'спинору бобка',
    'вафля творенном', 'всадилось чтете', 'сивым кульки', 'подмахнёт винчи',
    'шатнув зубиле', 'лесная тропа', 'старый город', 'свежий ветер',
    'белая птица', 'новая книга', 'синяя волна', 'чистое поле', 'тихий вечер',
    'добрый ветер', 'тёмная вода', 'быстрый ручей', 'серый камень',
    'светлый берег', 'тонкая нить', 'яркая звезда', 'зимняя дорога',
    'летняя трава', 'мягкий ветер', 'тёплый дождь', 'холодный день',
]

def available_fonts(font_dir=None):
    directory = Path(font_dir) if font_dir else Path('C:/Windows/Fonts')
    candidates = [('times.ttf', 4), ('timesbd.ttf', 4), ('arialbd.ttf', 2),
                  ('georgia.ttf', 1), ('cambria.ttc', 1)]
    result = [(directory / n, w) for n, w in candidates if (directory / n).is_file()]
    if not result:
        raise ValueError(f'No serif fonts in {directory}; use --font-dir.')
    return result

def straight_mask(text, font_path, size, rng=None):
    font = ImageFont.truetype(str(font_path), size * SCALE)
    box = font.getbbox(text, anchor='ls')
    pad = 10 * SCALE
    mask = Image.new('L', (box[2]-box[0]+pad*2, box[3]-box[1]+pad*2))
    draw = ImageDraw.Draw(mask)
    if rng is None:
        draw.text((pad-box[0], pad-box[1]), text, font=font, anchor='ls', fill=255)
    else:
        # Keep font advances and a shared baseline; add bounded per-glyph offsets.
        # A letter and its accent are drawn together, so accents cannot detach.
        strength = rng.uniform(1.7,4.0)*SCALE
        for i,ch in enumerate(text):
            x = pad-box[0]+font.getlength(text[:i+1])-font.getlength(ch)
            offset = rng.uniform(-strength,strength) if ch != ' ' else 0
            draw.text((x,pad-box[1]+offset),ch,font=font,anchor='ls',fill=255)
    return mask, pad-box[1]

def curve_mask(source, baseline, half_angle, direction):
    """Inverse annular mapping preserves the complete line's baseline."""
    src = np.asarray(source, dtype=np.float32)
    height, width = src.shape
    radius = max(width / (2 * half_angle), height * 1.6)
    half_angle = width / (2 * radius)
    # Restrict inverse sampling to the annular segment's actual bounds.
    radii = radius-direction*(np.array([0,height-1])-baseline)
    angles = np.array([-half_angle,0,half_angle])
    bx = radii[:,None]*np.sin(angles)
    by = direction*(radius-radii[:,None]*np.cos(angles))
    xmin,xmax = math.floor(bx.min())-2,math.ceil(bx.max())+2
    ymin,ymax = math.floor(by.min())-2,math.ceil(by.max())+2
    yy, xx = np.mgrid[ymin:ymax+1, xmin:xmax+1]
    radial_y = radius - direction * yy
    theta = np.arctan2(xx, radial_y)
    sx = radius * theta + (width-1)/2
    sy = baseline + direction * (radius - np.hypot(xx, radial_y))
    valid = (sx >= 0) & (sx < width-1) & (sy >= 0) & (sy < height-1)
    x0 = np.clip(np.floor(sx).astype(int), 0, width-2)
    y0 = np.clip(np.floor(sy).astype(int), 0, height-2)
    fx, fy = sx-x0, sy-y0
    values = ((1-fx)*(1-fy)*src[y0,x0] + fx*(1-fy)*src[y0,x0+1]
              + (1-fx)*fy*src[y0+1,x0] + fx*fy*src[y0+1,x0+1])
    output = Image.fromarray(np.where(valid, values, 0).clip(0,255).astype('uint8'))
    bounds = output.getbbox()
    if bounds is None:
        raise ValueError('Text has no visible glyphs.')
    return output.crop(bounds)

def text_layer(text, fonts, rng):
    if not text.strip() or len(text) > 48 or any(c in text for c in '\t\r\n'):
        raise ValueError('Use a nonempty single line of at most 48 characters without tabs.')
    font_path = rng.choices([p for p,w in fonts], [w for p,w in fonts])[0]
    direction = rng.choice([-1, 1])
    half_angle = rng.uniform(1.22, 1.48)
    rotation = rng.uniform(-9, 9)
    size = rng.randint(29,35)
    straight, baseline = straight_mask(text, font_path, size, rng)
    curved = curve_mask(straight, baseline, half_angle, direction)
    curved = curved.rotate(rotation, Image.Resampling.BICUBIC, expand=True)
    curved = curved.crop(curved.getbbox())
    target_w, target_h = rng.randint(158,176)*SCALE, rng.randint(75,81)*SCALE
    # The references use tall, condensed glyphs. Scale the complete line,
    # never individual characters, into the measured reference envelope.
    curved = curved.resize((target_w, target_h), Image.Resampling.LANCZOS)
    layer = Image.new('L', (FINAL_W*SCALE, FINAL_H*SCALE))
    px = (layer.width-curved.width)//2 + rng.randint(-4,4)*SCALE
    py = (layer.height-curved.height)//2 + rng.randint(-2,2)*SCALE
    layer.paste(curved, (px,py))
    return layer, {'font':font_path.name, 'direction':direction,
        'half_angle_deg':round(math.degrees(half_angle),3),
        'rotation_deg':round(rotation,3), 'font_size':size,
        'text_bbox':[round(v/SCALE,3) for v in layer.getbbox()]}

def make_background(rng):
    np_rng = np.random.default_rng(rng.getrandbits(64))
    w,h = FINAL_W*SCALE, FINAL_H*SCALE
    field = np.zeros((h,w), dtype=np.float32)
    for cols,rows,amplitude in [(6,4,24),(11,6,8)]:
        coarse = np_rng.uniform(0,255,(rows,cols)).astype('uint8')
        smooth = Image.fromarray(coarse).resize((w,h), Image.Resampling.BICUBIC)
        smooth = smooth.filter(ImageFilter.GaussianBlur(2.2*SCALE))
        field += (np.asarray(smooth,dtype=np.float32)-127.5)*amplitude/50
    base = rng.uniform(137,157)
    field += base
    regions = Image.new('L', (w,h), 0)
    middle = rng.uniform(25,65)*SCALE
    slant = rng.uniform(45,110)*SCALE*rng.choice([-1,1])
    left, right = middle-slant/2, middle+slant/2
    if rng.random() < .85:
        edge = h if rng.random() < .5 else 0
        polygon = [(0,left),(w,right),(w,edge),(0,edge)]
    else:
        thickness = rng.uniform(35,65)*SCALE
        polygon = [(0,left),(w,right),(w,right+thickness),(0,left+thickness)]
    ImageDraw.Draw(regions).polygon(polygon, fill=255)
    regions = regions.filter(ImageFilter.GaussianBlur(rng.uniform(.1,.55)*SCALE))
    delta = -rng.uniform(30,48)
    field += np.asarray(regions,dtype=np.float32)/255*delta
    # Additional translucent bands/lines vary independently of the main edge.
    line_count = rng.randint(1,4)
    for _ in range(line_count):
        lines = Image.new('L',(w,h),0)
        y0 = rng.uniform(-20,110)*SCALE
        y1 = y0+rng.uniform(-100,100)*SCALE
        line_width = rng.uniform(.7,7.0)*SCALE
        ImageDraw.Draw(lines).line([(0,y0),(w,y1)],fill=255,
                                   width=max(1,round(line_width)))
        lines = lines.filter(ImageFilter.GaussianBlur(rng.uniform(.2,1.2)*SCALE))
        field += np.asarray(lines,dtype=np.float32)/255*rng.uniform(-27,21)
    field += np_rng.normal(0,rng.uniform(.7,2),(h,w))
    return Image.fromarray(np.clip(field,45,218).astype('uint8')), {
        'background_base':round(base,3), 'region_delta':round(delta,3),
        'background_lines':line_count}

def generate_sample(text, fonts, rng):
    mask, metadata = text_layer(text,fonts,rng)
    bg, bg_metadata = make_background(rng)
    ink = rng.randint(7,24)
    image = Image.composite(Image.new('L',bg.size,ink), bg, mask)
    image = image.resize((FINAL_W,FINAL_H), Image.Resampling.LANCZOS)
    blur = rng.uniform(.28,.53)
    image = image.filter(ImageFilter.GaussianBlur(blur))
    grain_rng = np.random.default_rng(rng.getrandbits(64))
    grain = rng.uniform(2.3,4.8)
    pixels = np.asarray(image,dtype=np.float32)
    image = Image.fromarray(np.clip(pixels+grain_rng.normal(0,grain,pixels.shape),
                                    0,255).astype('uint8'))
    metadata.update(bg_metadata, ink=ink, blur=round(blur,3), grain=round(grain,3), text=text)
    return image.convert('RGB'),metadata,mask

def generate_image(text, fonts, seed=None):
    image,metadata,_ = generate_sample(text,fonts,random.Random(seed))
    return image,metadata

def load_phrases(path):
    if path is None:
        return DEFAULT_PHRASES
    phrases = [s.strip() for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if not phrases:
        raise ValueError('The phrases file is empty.')
    if any('\t' in p or len(p)>48 for p in phrases):
        raise ValueError('Phrases must have no tabs and at most 48 characters.')
    return phrases

def preview_sheet(output, reference_dir):
    paths = sorted((output/'images').glob('synthetic_*.jpg'))
    if len(paths) > 24:
        paths = [paths[i] for i in np.linspace(0,len(paths)-1,24,dtype=int)]
    refs = sorted(reference_dir.glob('*.jpg'))[:8] if reference_dir.is_dir() else []
    font = ImageFont.load_default(size=15)
    rows = max(len(refs),math.ceil(len(paths)/3))
    sheet = Image.new('RGB',(1040,38+rows*125),'#eeeae5')
    draw = ImageDraw.Draw(sheet)
    draw.text((10,10),'Reference originals',fill='black',font=font)
    draw.text((270,10),'Generated v4 / independent procedural images',fill='black',font=font)
    for i,path in enumerate(refs):
        with Image.open(path) as im:
            sheet.paste(im.convert('RGB'),(10,38+i*125))
        draw.text((10,130+i*125),path.stem[-8:],fill='black',font=font)
    for i,path in enumerate(paths):
        x,y = 270+(i%3)*255,38+(i//3)*125
        with Image.open(path) as im:
            sheet.paste(im.convert('RGB'),(x,y))
        draw.text((x,y+92),path.stem,fill='black',font=font)
    sheet.save(output/'comparison.jpg',quality=95)

def generate_dataset(count, output, phrases, seed=None, font_dir=None, overwrite=False, preview=True):
    if count < 1:
        raise ValueError('Count must be positive.')
    fonts = available_fonts(font_dir)
    if not phrases or any(not p.strip() or len(p)>48 or any(c in p for c in '\t\r\n') for p in phrases):
        raise ValueError('Invalid phrases: use nonempty lines of at most 48 characters.')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise ValueError(f'Output is not empty: {output}. Choose another folder.')
        existing = sorted((output/'images').glob('synthetic_*.jpg'))
        if [p.name for p in existing] != [f'synthetic_{i:06d}.jpg' for i in range(count)]:
            raise ValueError('Overwrite requires the same image count; choose a new folder.')
    image_dir = output/'images'
    image_dir.mkdir(parents=True,exist_ok=True)
    seed = seed if seed is not None else random.SystemRandom().getrandbits(63)
    rng = random.Random(seed)
    with (output/'labels.tsv').open('w',encoding='utf-8',newline='') as lf, (output/'metadata.jsonl').open('w',encoding='utf-8') as mf:
        labels = csv.writer(lf,delimiter='\t',lineterminator='\n')
        for i in range(count):
            text = rng.choice(phrases)
            sample_seed = rng.getrandbits(63)
            sample_rng = random.Random(sample_seed)
            image,metadata,_ = generate_sample(text,fonts,sample_rng)
            name = f'synthetic_{i:06d}.jpg'
            quality = sample_rng.randint(79,91)
            image.save(image_dir/name,quality=quality,subsampling=0)
            labels.writerow([f'images/{name}',text])
            metadata.update(file=name,sample_seed=sample_seed,jpeg_quality=quality)
            mf.write(json.dumps(metadata,ensure_ascii=False)+'\n')
            if (i+1)%100 == 0 or i+1 == count:
                print(f'{i+1}/{count}',flush=True)
    (output/'run.json').write_text(json.dumps({'generator_version':4,'seed':seed,
        'count':count,'size':[FINAL_W,FINAL_H],'phrases':phrases,
        'dependencies':{'Pillow':PILLOW_VERSION,'numpy':np.__version__},
        'fonts':[str(p) for p,w in fonts],
        'font_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p,w in fonts}},ensure_ascii=False,indent=2),encoding='utf-8')
    if preview:
        preview_sheet(output,ROOT/'test_samples')
    print(f'Saved: {output}\nSeed: {seed}')
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=30)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--phrases-file',type=Path)
    parser.add_argument('--font-dir',type=Path)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--no-preview',action='store_true')
    args = parser.parse_args()
    try:
        generate_dataset(args.count,args.output,load_phrases(args.phrases_file),
                         args.seed,args.font_dir,args.overwrite,not args.no_preview)
    except (ValueError,OSError) as exc:
        parser.exit(2,f'Error: {exc}\n')

if __name__ == '__main__':
    main()
