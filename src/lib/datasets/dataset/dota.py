# -*- encoding:utf-8 -*-
# @Time    : 2019/10/14 18:12
# @Author  : gfjiang
# @Site    : 
# @File    : dota.py
# @Software: PyCharm
import os
import os.path as osp
import numpy as np
import pycocotools.coco as coco
from tqdm import tqdm
import mmcv
from collections import defaultdict
from multiprocessing import Pool
import polyiou
import cvtools

from .coco import COCO


class DOTA(COCO):
    num_classes = 15
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(COCO, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'DOTA')
        self.img_dir = os.path.join(self.data_dir, 'crop')
        if split == 'test':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'crop800x800/val_dota+crop800x800.json').format(split)
        else:
            if opt.task == 'exdet':
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    'instances_extreme_{}2017.json').format(split)
            else:
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    'crop800x800/{}_dota+crop800x800.json').format(split)
        self.max_objs = 500
        self.class_name = [
            '__background__', 'large-vehicle', 'swimming-pool', 'helicopter',
            'bridge', 'plane', 'ship', 'soccer-ball-field', 'basketball-court',
            'ground-track-field', 'small-vehicle', 'harbor', 'baseball-diamond',
            'tennis-court', 'roundabout', 'storage-tank']
        self._valid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        print('==> initializing coco 2017 {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def crop_bbox_map_back(self, bb, crop_start):
        bb_shape = bb.shape
        original_bb = bb.reshape(-1, 2) + np.array(crop_start).reshape(-1, 2)
        return original_bb.reshape(bb_shape)

    def genereteImgResults(self, results):
        """结合子图的结果，映射回原图，应用nms, 生成一张整图的结果"""
        imgResults = defaultdict(list)
        for image_id, dets in results.items():
            img_info = self.coco.imgs[image_id]
            labels = mmcv.concat_list([[j]*len(det)for j, det in dets.items()])
            scores = mmcv.concat_list([det[:, 8] for det in dets.values()])
            rbboxes = np.vstack([det[:, :8] for det in dets.values() if len(det) > 0])
            if 'crop' in img_info:
                rbboxes = self.crop_bbox_map_back(rbboxes, img_info['crop'][:2])
            assert len(rbboxes) == len(labels)
            if len(labels) > 0:
                result = [rbboxes, labels, scores]
                imgResults[img_info['file_name']].append(result)
        return imgResults

    def py_cpu_nms_poly(self, dets, thresh):
        scores = dets[:, 8]
        polys = []
        for i in range(len(dets)):
            tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                               dets[i][2], dets[i][3],
                                               dets[i][4], dets[i][5],
                                               dets[i][6], dets[i][7]])
            polys.append(tm_polygon)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            ovr = []
            i = order[0]
            keep.append(i)
            for j in range(order.size - 1):
                iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
                ovr.append(iou)
            ovr = np.array(ovr)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def merge_results(self, anns, results, n_worker=0):
        # anns = mmcv.load(ann_file)
        # results = mmcv.load(result_file)
        imgResults = defaultdict(list)
        if n_worker > 0:
            pool = Pool(processes=n_worker)
            num = len(anns) // n_worker
            anns_group = [anns[i:i + num] for i in range(0, len(anns), num)]
            results_group = [results[i:i + num] for i in range(0, len(results), num)]
            res = []
            for anns, results in tqdm(zip(anns_group, results_group)):
                res.append(pool.apply_async(self.genereteImgResults, args=(anns, results,)))
            pool.close()
            pool.join()
            for item in res:
                imgResults.update(item.get())
        else:
            imgResults = self.genereteImgResults(results)
        for filename, result in imgResults.items():
            rbboxes = np.vstack([bb[0] for bb in result]).astype(np.int)
            labels = np.hstack([bb[1] for bb in result])
            scores = np.hstack([bb[2] for bb in result])
            ids = self.py_cpu_nms_poly(np.hstack([rbboxes, scores[:, np.newaxis]]), 0.3)
            # rbboxes = np.hstack([rbboxes, labels, scores])
            imgResults[filename] = [rbboxes[ids], labels[ids], scores[ids]]
        return imgResults

    def ImgResults2CatResults(self, imgResults):
        catResults = defaultdict(list)
        for filename in imgResults:
            rbboxes = imgResults[filename][0]
            cats = imgResults[filename][1]
            scores = imgResults[filename][2]
            for ind, cat in enumerate(cats):
                catResults[cat].append([filename, scores[ind], rbboxes[ind]])
        return catResults

    def writeResults2DOTATestFormat(self, catResults, class_names, save_path):
        for cat_id, result in catResults.items():
            lines = []
            for filename, score, rbbox in result:
                filename = osp.splitext(filename)[0]
                bbox = list(map(str, list(rbbox)))
                score = str(round(score, 3))
                lines.append(' '.join([filename] + [score] + bbox))
            cvtools.write_list_to_file(
                lines, osp.join(save_path, 'Task1_' + class_names[cat_id] + '.txt'))

    def run_eval(self, results, save_dir):
        # self.save_results(results, save_dir)
        imgResults = self.merge_results(self.coco.anns, results)
        catResults = self.ImgResults2CatResults(imgResults)
        self.writeResults2DOTATestFormat(catResults, self.class_name, self.opt.save_dir+'/DOTA_results')
        os.system("/opt/conda/bin/python /root/DOTA_devkit/dota_evaluation_task1.py")
