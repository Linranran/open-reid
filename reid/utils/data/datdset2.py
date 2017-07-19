from __future__ import print_function
import os.path as osp

import numpy as np

from ..serialization import read_json

def _pluck(identities, indices, relabel=False):                  
ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                if relabel:
                    ret.append((fname,camid, index ))
                else:
                    ret.append((fname, pid, camid))
    return ret







class Dataset2(object):
    def __init__(self, root, split_id=0):
        self.root = root
        self.split_id = split_id
        self.meta = None
        self.split = None
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    def load(self, num_val=0.3, verbose=True):


         



        def take (m,n):
            for i in range(m,n):
                yield i



        train_pids=list(take(0,221599))
        
        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']
        
        self.wwwww = _pluck(identities, train_pids, relabel=True)
        
