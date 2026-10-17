"""Functional validation of generation, bounds and dataset integrity."""
import contextlib
import csv
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
from PIL import Image
import generator as g

class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fonts = g.available_fonts()

    def test_reproducible_and_varied(self):
        a,ma = g.generate_image('подмахнёт винчи',self.fonts,918)
        b,mb = g.generate_image('подмахнёт винчи',self.fonts,918)
        c,mc = g.generate_image('подмахнёт винчи',self.fonts,919)
        self.assertEqual(a.tobytes(),b.tobytes())
        self.assertEqual(ma,mb)
        self.assertNotEqual(a.tobytes(),c.tobytes())
        self.assertEqual(a.size,(250,90))
        self.assertEqual(a.mode,'RGB')

    def test_many_seeds_have_complete_safe_text_bounds(self):
        seen = set()
        for seed in range(120):
            text = g.DEFAULT_PHRASES[seed % len(g.DEFAULT_PHRASES)]
            mask,meta = g.text_layer(text,self.fonts,random.Random(seed))
            a = np.asarray(mask)
            self.assertFalse(a[:6].any())
            self.assertFalse(a[-6:].any())
            self.assertFalse(a[:,:6].any())
            self.assertFalse(a[:,-6:].any())
            x0,y0,x1,y1=mask.getbbox()
            self.assertGreater(x1-x0,150*g.SCALE)
            self.assertGreater(y1-y0,70*g.SCALE)
            seen.add(meta['direction'])
        self.assertEqual(seen,{-1,1})

    def test_accents_descenders_and_length_limits(self):
        for text in ['й ё щ ц р у', 'й', 'ш'*48, 'Щёлкнувший ёж']:
            image,meta,mask = g.generate_sample(text,self.fonts,random.Random(42))
            self.assertIsNotNone(mask.getbbox())
            self.assertEqual(meta['text'],text)
        for text in ['', '   ', 'text\tlabel', 'x'*49, 'two\nlines']:
            with self.assertRaises(ValueError):
                g.generate_image(text,self.fonts,1)

    def test_curve_direction(self):
        # Repeated narrow strokes make curvature measurable independently of words.
        source,baseline = g.straight_mask('llllllllllllllll',self.fonts[0][0],32)
        for direction in [-1,1]:
            mask = np.asarray(g.curve_mask(source,baseline,1.4,direction))
            yy,xx=np.where(mask>127)
            lo,hi=np.quantile(xx,[.2,.8])
            center=np.median(yy[(xx>np.quantile(xx,.4))&(xx<np.quantile(xx,.6))])
            ends=np.median(yy[(xx<lo)|(xx>hi)])
            self.assertGreater((ends-center)*direction,10)

    def test_dataset_labels_replay_and_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'dataset'
            with contextlib.redirect_stdout(io.StringIO()):
                g.generate_dataset(5,out,['подмахнёт винчи','сивым кульки'],18,preview=False)
            with (out/'labels.tsv').open(encoding='utf-8') as label_file:
                labels=list(csv.reader(label_file,delimiter='\t'))
            metadata=[json.loads(s) for s in (out/'metadata.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(labels),5)
            for (path,text),meta in zip(labels,metadata):
                self.assertEqual(text,meta['text'])
                actual=(out/path).read_bytes()
                im,_=g.generate_image(text,self.fonts,meta['sample_seed'])
                buf=io.BytesIO()
                im.save(buf,format='JPEG',quality=meta['jpeg_quality'],subsampling=0)
                self.assertEqual(actual,buf.getvalue())
            before=(out/'labels.tsv').read_bytes()
            with self.assertRaises(ValueError):
                g.generate_dataset(5,out,['other'],18,preview=False)
            with self.assertRaises(ValueError):
                g.generate_dataset(6,out,['other'],18,overwrite=True,preview=False)
            self.assertEqual(before,(out/'labels.tsv').read_bytes())
            with contextlib.redirect_stdout(io.StringIO()):
                g.generate_dataset(5,out,['подмахнёт винчи','сивым кульки'],18,overwrite=True,preview=False)
            self.assertEqual(before,(out/'labels.tsv').read_bytes())

if __name__ == '__main__':
    unittest.main(verbosity=2)
