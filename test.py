import os
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import mmcv
from collections import Counter
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import cv2

config_file = r'./configs/convnext/convnext_b_test.py'
checkpoint_file = r"./checkpoints/latest.pth"
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
img_root = r'./samples/img/'
save_mask_root = r'./results/'

if not os.path.exists(save_mask_root):
    os.mkdir(save_mask_root)
img_names = os.listdir(img_root)
for img_name in tqdm(img_names):
    img = img_root + img_name
    mask_name=img_name[:-4] + '.png'
    path = os.path.join(save_mask_root, mask_name)
    result = inference_segmentor(model, img)[0]
    img = Image.fromarray(np.uint8(result*255))
    img.save(save_mask_root + mask_name)

def metric(premask, groundtruth):
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = np.logical_and(premask, groundtruth).sum().astype(np.float64)
    true_neg = np.logical_and(seg_inv, gt_inv).sum().astype(np.float64)
    false_pos = np.logical_and(premask, gt_inv).sum().astype(np.float64)
    false_neg = np.logical_and(seg_inv, groundtruth).sum().astype(np.float64)
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    if np.sum(cross) + np.sum(union) == 0:
        iou = 1
    return f1, iou

path_gt = './samples/gt/'
if os.path.exists(path_gt):
    flist = sorted(os.listdir(save_mask_root))
    auc, f1, iou = [], [], []
    for file in tqdm(flist):
        pre = cv2.imread(save_mask_root + file)
        gt = cv2.imread(path_gt + file[:-4] + '.png')
        H, W, C = pre.shape
        Hg, Wg, C = gt.shape
        if H != Hg or W != Wg:
            gt = cv2.resize(gt, (W, H))
            gt[gt > 127] = 255
            gt[gt <= 127] = 0
        if np.max(gt) != np.min(gt):
            auc_t = roc_auc_score((gt.reshape(H * W * C) / 255).astype('int'), pre.reshape(H * W * C) / 255.)
            auc.append(auc_t)
        pre[pre > 127] = 255
        pre[pre <= 127] = 0
        a, b = metric(pre / 255, gt / 255)
        f1.append(a)
        iou.append(b)
    print('Evaluation: AUC: %5.4f, F1: %5.4f, IOU: %5.4f' % (np.mean(auc), np.mean(f1), np.mean(iou)))
