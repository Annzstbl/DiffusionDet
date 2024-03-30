#引入以下注释
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import pycocotools
import os
import cv2
from detectron2.utils.visualizer import ColorMode, Visualizer

#声明类别，尽量保持
CLASS_NAMES =["people","bike","worker","car"]
# 数据集路径
DATASET_ROOT = '/data3/litianhao/datasets/HOD3K/hsidetection/sa_information/'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
IMG_ROOT = os.path.join(DATASET_ROOT, 'images')
 
TRAIN_PATH = os.path.join(IMG_ROOT, 'train')
VAL_PATH = os.path.join(IMG_ROOT, 'val')
TEST_PATH = os.path.join(IMG_ROOT, 'test')
 
TRAIN_JSON = os.path.join(ANN_ROOT, 'train.json')
VAL_JSON = os.path.join(ANN_ROOT, 'val.json')
TEST_JSON = os.path.join(ANN_ROOT, 'test.json')
 
# 声明数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "hod3k_sa_train": (TRAIN_PATH, TRAIN_JSON),
    "hod3k_sa_val": (VAL_PATH, VAL_JSON),
}
#===========以下有两种注册数据集的方法，本人直接用的第二个plain_register_dataset的方式 也可以用register_dataset的形式==================
#注册数据集（这一步就是将自定义数据集注册进Detectron2）
def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   json_file=json_file,
                                   image_root=image_root)
 
 
#注册数据集实例，加载数据集中的对象实例
def register_dataset_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco")
 
#=============================
# 注册数据集和元数据
def plain_register_dataset():
    #训练集
    DatasetCatalog.register("hod3k_sa_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("hod3k_sa_train").set(thing_classes=CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                    evaluator_type='coco', # 指定评估方式
                                                    json_file=TRAIN_JSON,
                                                    image_root=TRAIN_PATH)
 
    #DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
    #验证/测试集
    DatasetCatalog.register("hod3k_sa_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("hod3k_sa_val").set(thing_classes=CLASS_NAMES, # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                evaluator_type='coco', # 指定评估方式
                                                json_file=VAL_JSON,
                                                image_root=VAL_PATH)
# 查看数据集标注，可视化检查数据集标注是否正确，
#这个也可以自己写脚本判断，其实就是判断标注框是否超越图像边界
#可选择使用此方法
def checkout_dataset_annotation(name="hod3k_sa_train", save_path="out"):
    #dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
    dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH)
    print(len(dataset_dicts))
    for i, d in enumerate(dataset_dicts,0):
        #print(d)
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
        vis = visualizer.draw_dataset_dict(d)
        #cv2.imshow('show', vis.get_image()[:, :, ::-1])
        img_name = str(i) + '.jpg'
        cv2.imwrite(os.path.join(save_path, img_name), vis.get_image()[:, :, ::-1])
        #cv2.waitKey(0)
        if i == 200:
            break


def file_test():
    register_dataset()
    checkout_dataset_annotation()

if __name__ == '__main__':
    #file_test()
    plain_register_dataset()
    checkout_dataset_annotation()