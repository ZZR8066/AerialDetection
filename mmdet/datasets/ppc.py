from .coco import CocoDataset


class PPCDataset(CocoDataset):

    CLASSES = (
            'Pens', 'Pencils', 'Chopsticks'      
    )