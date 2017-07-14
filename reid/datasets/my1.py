from __future__ import print_function, absolute_import
import os.path as osp

import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class MY1(Dataset):
    def __init__(self, root, split_id=0, num_val=100, download=False):
        super(MY1, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")
        self.load(num_val)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return


###### ..............................................................MARK1501


        from six.moves import urllib

        import re

        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile
        from scipy.misc import imsave, imread
        import h5py
        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'Market-1501-v15.09.15.zip')
        # if osp.isfile(fpath) and \
        #   hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
        #     print("Using downloaded file: " + fpath)
        # else:
        #     raise RuntimeError("Please download the dataset manually from {} "
        #                        "to {}".format(self.url, fpath))

        # Extract the file
        exdir = osp.join(raw_dir, 'Market-1501-v15.09.15')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        # 1501 identities (+1 for background) with 6 camera views each
        identities = [[[] for _ in range(6)] for _ in range(1502)]

        def register(subdir, pattern=re.compile(r'([-\d]+)_c(\d)')):
            fpaths = sorted(glob(osp.join(exdir, subdir, '*.jpg')))
            pids = set()
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid, cam = map(int, pattern.search(fname).groups())
                if pid == -1: continue  # junk images are just ignored
                assert 0 <= pid <= 1501  # pid == 0 means background
                assert 1 <= cam <= 6
                cam -= 1
                pids.add(pid)
                fname = ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                shutil.copy(fpath, osp.join(images_dir, fname))
            return pids
#....................................................................cukn03

        # Extract the file
        fpath = osp.join(raw_dir, 'cuhk03_release.zip')
        exdir = osp.join(raw_dir, 'cuhk03_release')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)
        matdata = h5py.File(osp.join(exdir, 'cuhk-03.mat'), 'r')

        def deref(ref):
            return matdata[ref][:].T

        def dump_(refs, pid, cam, fnames):
            for ref in refs:
                img = deref(ref)
                if img.size == 0 or img.ndim < 2: break
                fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid+1501, cam, len(fnames))
                imsave(osp.join(images_dir, fname), img)
                fnames.append(fname)

        #identities = []
        for labeled, detected in zip(
                matdata['labeled'][0], matdata['detected'][0]):
            labeled, detected = deref(labeled), deref(detected)
            assert labeled.shape == detected.shape
            for i in range(labeled.shape[0]):
                pid = len(identities)
                images = [[], []]
                dump_(labeled[i, :5], pid, 0, images[0])
                dump_(detected[i, :5], pid, 0, images[0])
                dump_(labeled[i, 5:], pid, 1, images[1])
                dump_(detected[i, 5:], pid, 1, images[1])
                identities.append(images)
#............................................viper

        fpath = osp.join(raw_dir, 'VIPeR.v1.0.zip')
        exdir = osp.join(raw_dir, 'VIPeR')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)

        #Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)
        cameras = [sorted(glob(osp.join(exdir, 'cam_a', '*.bmp'))),
                   sorted(glob(osp.join(exdir, 'cam_b', '*.bmp')))]
        assert len(cameras[0]) == len(cameras[1])
        # identities = []
        for pid, (cam1, cam2) in enumerate(zip(*cameras)):
            images = []
            # view-0
            # fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 0, 0)
            fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid+2968, 0, 0)
            imsave(osp.join(images_dir, fname), imread(cam1))
            images.append([fname])
            # view-1
            fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid+2968, 1, 0)
            imsave(osp.join(images_dir, fname), imread(cam2))
            images.append([fname])
            identities.append(images)
#............................
        fpath = osp.join(raw_dir, 'CUHK01.zip')
        # Extract the file
        exdir = osp.join(raw_dir, 'campus')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        identities1 = [[[] for _ in range(2)] for _ in range(971)]

        files = sorted(glob(osp.join(exdir, '*.png')))
        for fpath in files:
            fname = osp.basename(fpath)
            pid, cam = int(fname[:4]), int(fname[4:7])
            assert 1 <= pid <= 971
            assert 1 <= cam <= 4
            pid, cam = pid - 1, (cam - 1) // 2
            fname = ('{:08d}_{:02d}_{:04d}.png'
                     .format(pid+3590, cam, len(identities1[pid][cam])))
            identities1[pid][cam].append(fname)
            shutil.copy(fpath, osp.join(images_dir, fname))


        for a1 in identities1:
            identities.append(a1)

#...........................................
        meta = {'name': 'MY', 'shot': 'multiple', 'num_cameras': 10,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Randomly create ten training and test split
        num = len(identities)
        splits = []
        for _ in range(10):
            pids = np.random.permutation(num).tolist()
            trainval_pids = sorted(pids[:num // 2])
            test_pids = sorted(pids[num // 2:])
            split = {'trainval': trainval_pids,
                     'query': test_pids,
                     'gallery': test_pids}
            splits.append(split)
        write_json(splits, osp.join(self.root, 'splits.json'))
