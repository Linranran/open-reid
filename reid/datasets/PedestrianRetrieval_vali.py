from __future__ import print_function, absolute_import
import os.path as osp

import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class PedestrianRetrieval_vali(Dataset):
    def __init__(self, root, split_id=0, num_val=0.3, download=False):
        super(PedestrianRetrieval_vali, self).__init__(root, split_id=split_id)

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

        import hashlib
        from glob import glob
        from scipy.misc import imsave, imread
        from six.moves import urllib
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        # fpath = osp.join(raw_dir, 'VIPeR.v1.0.zip')
        # if osp.isfile(fpath) and \
        #    hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
        #     print("Using downloaded file: " + fpath)
        # else:
        #     print("Downloading {} to {}".format(self.url, fpath))
        #     urllib.request.urlretrieve(self.url, fpath)


        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)
        exdir = osp.join(raw_dir, 'a')
        files = sorted(glob(osp.join(exdir, '*.jpg')))

#..........................chuli tipian
        identities = []
        import shutil
        import operator
        import xml.etree.ElementTree as ET
        tree = ET.ElementTree(file='/home/lin/re_id/open-reid-master_1/examples/data/PedestrianRetrieval_vali/vali.xml')

        a = []
        for elem in tree.iter(tag='Item'):
            a.append(elem.attrib)

        sorted_a = sorted(a, key=operator.itemgetter('pedestrianID'))
        print(sorted_a)
        m = 0
        l = 0
        # o, file in enumerate(zip(*files))

        pedestrianID1 = 100004
        for i, j in enumerate(sorted_a):
            # for key in j[0]:
            imageName = j['imageName']
            pedestrianID = j['pedestrianID']
            aa = sorted(glob(osp.join(exdir, imageName)))
            images = []
            for imageName1 in files:
                if [imageName1]==aa :
                    break

            if pedestrianID1 == pedestrianID:
                pedestrianID1 = pedestrianID
                m = m + 1
                fname = '{:08d}_{:02d}_{:04d}.jpg'.format(l, 0, m)
            else:
                pedestrianID1 = pedestrianID
                l = l + 1
                m=0
                fname = '{:08d}_{:02d}_{:04d}.jpg'.format(l, 0, m)
            images.append([fname])
            identities.append(images)
            shutil.copy(imageName1, osp.join(images_dir, fname))






        # Save meta information into a json file
        meta = {'name': 'VIPeR', 'shot': 'single', 'num_cameras': 2,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Randomly create ten training and test split
        num = len(identities)
        pids = np.random.permutation(num).tolist()
        trainval_pids = sorted(pids[:num // 2])
        test_pids = sorted(pids[num // 2:])
        splits = [{'trainval': trainval_pids,
                  'query': test_pids,
                  'gallery': test_pids}]
        write_json(splits, osp.join(self.root, 'splits.json'))




