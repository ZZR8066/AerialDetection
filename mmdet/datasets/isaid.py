from .coco import CocoDataset
import numpy as np

import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset

class iSAIDDataset(CocoDataset):
        CLASSES = ('Small_Vehicle', 'Large_Vehicle', 'plane', 'storage_tank', 'ship', 'Swimming_pool', 'Harbor',         
                'tennis_court', 'Ground_Track_Field', 'Soccer_ball_field', 'baseball_diamond', 
                'Bridge', 'basketball_court', 'Roundabout', 'Helicopter')

        def load_annotations(self, ann_file, balance_cat):
                self.coco = COCO(ann_file)
                self.cat_ids = self.coco.getCatIds()
                self.cat2label = {
                cat_id: i + 1
                for i, cat_id in enumerate(self.cat_ids)
                }
                if balance_cat:
                    self.img_ids = enhance_dataset(self.coco)
                else:
                    self.img_ids = self.coco.getImgIds()
                img_infos = []
                for i in self.img_ids:
                        info = self.coco.loadImgs([i])[0]
                        info['filename'] = info['file_name']
                        img_infos.append(info)
                return img_infos


    
      




def concatenate_list(c):
    t = c[0]
    for i in range(1,len(c)):
        t = np.concatenate((t,c[i]),axis=0)
    return list(t)

def enhance_dataset(coco):
    imgIds = coco.getImgIds()
    imgs = coco.loadImgs(imgIds)
    cats = coco.loadCats(coco.getCatIds())
    classNum = len(cats)
    print('类别数：',classNum)
    # 对图片进行类别分类
    classfiy_result = [[] for _ in range(classNum)]
    classfiy_count  = np.zeros((classNum,),dtype=np.int)
    for img in imgs:
        if len(coco.getAnnIds(imgIds=img['id'])) == 0:
            pass
        else:
            # 统计每张图类别情况
            catNum = np.zeros((classNum,),dtype=np.int)
            for cat in cats:
                catId = cat['id']
                annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None,catIds=catId)
                catNum[catId-1] = catNum[catId-1] + len(annIds)
            ind = np.argmax(catNum)
            classfiy_result[ind].append(img['id'])
            classfiy_count[ind] = classfiy_count[ind] + 1
    print('原始数据集统计情况：')
    print(classfiy_count)
    # 计算每个图片增强系数
    coff  = np.zeros((classNum,),dtype=np.int)
    coff  = np.floor(classfiy_count.max() / classfiy_count)
    print('增强系数：')
    print(coff)
    # 进行数据增强
    enhance_result = [[] for _ in range(classNum)]
    for i in range(len(classfiy_result)):
        imgIds_ = classfiy_result[i]
        multiple = int(coff[i])
        for j in range(multiple):
            enhance_result[i].append(imgIds_)

    for i in range(len(enhance_result)):
        enhance_result[i] = concatenate_list(enhance_result[i])
    enhance_result = concatenate_list(enhance_result)

    imgs = coco.loadImgs(enhance_result)
    # 对图片进行类别分类
    classfiy_result = [[] for _ in range(classNum)]
    classfiy_count  = np.zeros((classNum,),dtype=np.int)
    for img in imgs:
        # 统计每张图类别情况
        catNum = np.zeros((classNum,),dtype=np.int)
        for cat in cats:
            catId = cat['id']
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None,catIds=catId)
            catNum[catId-1] = catNum[catId-1] + len(annIds)
        ind = np.argmax(catNum)
        classfiy_result[ind].append(img['id'])
        classfiy_count[ind] = classfiy_count[ind] + 1
    print('数据增强后的统计情况：')
    print(classfiy_count)
    print(len(enhance_result))
    return enhance_result