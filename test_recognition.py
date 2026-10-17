"""Semantics and leakage checks for the improved recognizer."""
import csv,json,unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import recognition_model as r
from recognition_pipeline import choose,Recognizer
from recognition_transforms import widen_gap,adaptive_crop,clipped_ink_pixels

class RecognitionTests(unittest.TestCase):
    def probabilities(self,tokens):
        q=np.zeros((len(tokens),len(r.CHARACTERS)),dtype='float32')
        for t,ch in enumerate(tokens):q[t,r.CHARACTERS.index(ch)]=1
        return q
    def test_repeated_letters_require_blank_and_survive_it(self):
        self.assertEqual(r.decode(self.probabilities(['а','blank','а']),True)[0],'аа')
        self.assertEqual(r.decode(self.probabilities(['а','а']),True)[0],'а')
    def test_case_probability_merge_precedes_ctc_collapse(self):
        self.assertEqual(r.decode(self.probabilities(['А','а','blank','А']),True)[0],'аа')
    def test_spaces_and_yo_remain_meaningful(self):
        self.assertEqual(r.decode(self.probabilities(['ё',' ','е']),True)[0],'ё е')
    def test_beam_search_sums_blank_paths_for_repeated_letters(self):
        p=np.zeros((5,len(r.CHARACTERS)),dtype='float32')
        p[:,0]=.4;p[:,r.CHARACTERS.index('а')]=.6
        self.assertEqual(r.decode(p,True)[0],'а')
        self.assertEqual(r.decode_beam(p)[0],'аа')
        self.assertEqual(r.decode_beam(self.probabilities(['ё','blank','ё']))[0],'ёё')
    def test_preprocessing_cannot_drop_horizontal_content(self):
        im=np.full((64,400,3),120,dtype='uint8')
        im[:,0]=255;im[:,-1]=0
        for scale in [.7,1,1.6]:
            out=r.preprocess(im,scale)
            self.assertEqual(out.shape,(3,48,320))
            self.assertTrue(np.isfinite(out).all())
    @unittest.skipUnless((r.ROOT/'experiments/recognition_v2/train/labels.tsv').is_file(),
                         'Private corpus is not included in the distribution')
    def test_words_are_disjoint_across_actual_label_files(self):
        base=r.ROOT/'experiments/recognition_v2'
        sets={}
        for split,expected in [('train',6000),('dev',400),('final',1000),
                               ('train_stress',1500),('dev_stress',200),('final_stress',500),
                               ('final_fresh',1000),('final_stress_fresh',500)]:
            with (base/split/'labels.tsv').open(encoding='utf-8') as f:
                rows=list(csv.reader(f,delimiter='\t'))
            self.assertEqual(len(rows),expected)
            sets[split]=set(' '.join(text for path,text in rows).split())
            self.assertTrue(all((base/split/path).is_file() for path,text in rows))
        names=list(sets)
        for i,name in enumerate(names):
            for other in names[i+1:]:
                self.assertFalse(sets[name]&sets[other],(name,other))
    def test_spacing_transform_preserves_ink_and_rejects_regular_letter_gaps(self):
        im=np.full((64,160,3),150,dtype='uint8')
        for x in [10,25,40,100,115,130]:im[22:43,x:x+8]=10
        widened,changed=widen_gap(im)
        self.assertTrue(changed)
        self.assertGreater(widened.shape[1],im.shape[1])
        self.assertEqual(int((im[:,:,0]<30).sum()),int((widened[:,:,0]<30).sum()))
        regular=np.full_like(im,150)
        for x in range(10,140,15):regular[22:43,x:x+8]=10
        unchanged,changed=widen_gap(regular)
        self.assertFalse(changed)
        np.testing.assert_array_equal(regular,unchanged)
    def test_candidate_selection_never_uses_expected_answers(self):
        candidates=[('летня трава',.98),('летняя трава',.97),('летняя трава',.96)]
        self.assertEqual(choose(candidates,{'method':'vote'})[0],'летняя трава')
        self.assertEqual(choose(candidates,{'method':'longest_close','tolerance':.02})[0],'летняя трава')
    def test_adaptive_crop_retains_a_letter_below_the_regular_crop(self):
        im=np.full((64,160,3),150,dtype='uint8')
        for x in [10,25,40,55]:im[20:38,x:x+8]=10
        im[54:63,100:109]=10
        self.assertGreater(clipped_ink_pixels(im),0)
        cropped=adaptive_crop(im)
        self.assertEqual(int((cropped[:,:,0]<30).sum()),int((im[:,:,0]<30).sum()))
    def test_rescue_requires_supported_single_insertion_and_confidence(self):
        recognizer=Recognizer.__new__(Recognizer)
        recognizer.model=object();recognizer.rescue_model=object()
        recognizer.config={'variants':[{'crop':40,'constrained':True,'scale':1}],
            'selection':{'method':'confidence'},'insertion_rescue':{'threshold':0,'minimum_confidence':.6}}
        image=np.full((64,160,3),150,dtype='uint8')
        image[54:63,100:109]=10
        for proposed,confidence,expected in [('крот ток',.7,'крот ток'),
                ('крот ток',.5,'кот ток'),('крат ток',.9,'кот ток'),('kрот ток',.9,'кот ток')]:
            with patch.object(r,'predict',side_effect=[[('кот ток',.9)],[(proposed,confidence)]]):
                self.assertEqual(recognizer.predict_rectified([image])[0][0],expected)

if __name__=='__main__':unittest.main(verbosity=2)
