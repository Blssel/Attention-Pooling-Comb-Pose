#!/usr/env python
#coding:utf-8
import os
import numpy as np
import tensorflow as tf
import math
__author__='Zhiyu Yin'

_MPII_MAT_FILE='/home/yzy/dataset/MPII/mpii_human_pose_v1_u12_1.mat'
_NUM_SHARDS=20 # 表示
_NUM_JOINTS=16

def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[Values]))

def int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def image_to_tfexample(image_data,image_format,height,width,pose,action_label):  #其中pose是[x,y,is_vis,...]
  return tf.train.Example(features=tf.train.Freatures(feature={
                          'image/encoded':bytes_feature(image_data),
                          'iamge/format':bytes_feawture(image_format),
                          'image/class/pose':int64_feature([int(el) for el in pose]),
                          'image/class/action_label': int64_feature(action_label),
                          'image/height':int64_feature(height),
                          'image/width':int64_feature(width)}))
  
# 功能：生成tfrecord文件名  （返回一个全路径）
def _get_dataset_filename(dataset_dir,split_name,shard_id):
  output_filename='mpii_%s_%05d-of-%05d.tfrecord'% (split_name,shard_id,_NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)

class ImageReader(object):
  # 这里没有使用opencv或其他库，而是用的tf图像处理函数
  def __init__(self):
    self._decode_jpeg_data=tf.placeholder(dtype=tf.string)  # 这个string很重要，因为decode_jpeg函数必须接受该类型的tensor
    self._decode_jpeg=tf.image.decode_jpeg(self._decode_jpeg_data,channels=3)

  def read_image_dims(self,sess,image_data):
    image=self.decode_jpeg(sess,image_data)
    return image.shape[0], image.shape[1]
    
  def decode_jpeg(self,sess,image_data):
    image=sess.run(self._decode_jpeg,feed_dict={self._decode_jpeg_data:image_data})
      
    return image

# split_name表示当前split的名称   
# list_to_write保存某（mou）一个split的image_obj列表
def _convert_dataset(split_name, list_to_write, dataset_dir):
  #首先，计算每个tfrecord文件保存的图片数据数量
  num_per_shard=int(math.ceil(len(list_to_write)/float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    # 先定义ImageReader类来读取数据
    image_reader=ImageReader()
    # 启动sess
    with tf.Session('') as sess:
      for shard_id in range(_NUM_SHARDS):
        # 生成该批次tfrecord的文件名
        output_filename=_get_dataset_filename(dataset_dir,split_name,shard_id)
        
        with tf.python_io.TFRecodWriter(output_filename) as tfrecod_writer:
          start_ndx= shard_id*num_per_shard #该批次的起始和结束索引(左闭右开)
          end_ndx=min(start_ndx+num_per_shard, len(list_to_write))
          # 遍历该批次每个文件
          for i in range(start_ndx,end_ndx):
            sys.out.write('\r>> Converting image %d/%d shard %d'% (i+1,len(list_to_write),shard_id))
            sys.stdout.flush()

            # 读取文件
            fname=os.path.join(_IMG_DIR,list_to_write[i][0])
            action_label=list_to_write[i][1]
            poses=list_to_write[i][2]
            all_joints=[]
            for pose in poses:
              joints=dict((el[0],[el[1],el[2],el[3]]) for el in pose)
              final_pose=[]
              for i in range(_NUM_JOINTS):
                if i in joints:
                  final_pose.append(joints[i])
                else:
                  final_pose.append([-1,-1,0])
              final_pose=[item for sublist in]#??????????????

            image_data=tf.gfile.FastGFile(fname,'r').read()
            height,width=image_reader.read_image_dims(sess,image_data)
            
            example=image_to_tfexample(image_data,'jpg',height,width,all_joints,action_label) #注意这里的image_data数据是为解码的
            tfrecod_writer.write(example.SerializeToString())
  
  
def main():
  splits=['train','val','test']

  # 制作split的索引
  # 依据本地的三个split文件 生成splits_filenames filename_to_split
  _SPLITS_PATH = '../../src/data/mpii/lists/spli_{}.txt'
  splits_filenames={}
  filename_to_split={}
  for spl in splits:
    with open(_SPLITS_PATH.format(spl),'r') as fin:
      splits_filenames[spl]= fin.read().splitlines()
      filename_to_split.update(dict(zip(
                              splits_filenames[spl], [spl]*len(splits_filenames[spl]))))
  
  # 导入注释数据
  import scipy.io
  T=scipy.io.loadmat(_MPII_MAT_FILE,squeeze_me=True,struct_as_record=False)
  '''
  type(T['RELEASE'])  返回<class 'scipy.io.matlab.mio5_params.mat_struct'>
  type(T['RELEASE'].annolist) 返回<type 'numpy.ndarray'>
  type(T['RELEASE'].annolist[0])   返回<class 'scipy.io.matlab.mio5_params.mat_struct'>
  type(T['RELEASE'].annolist[0].image.name) 返回<type 'unicode'>
  type(T['RELEASE'].annolist[0].annorect)   返回<class 'scipy.io.matlab.mio5_params.mat_struct'>
  type(T['RELEASE'].annolist[0].annorect.annopoints.point  返回mat_struct' object has no attribute 'annopoints'  只有.scale和.objpos属性)
  '''
  all_imnames=[] # 保存所有图片的文件名
  annots=T['RELEASE'].annolist # 是个列表，其中每一个元素对应一张图片的注释（是个对象类型）
  for aid, annot in enumerate(annots):
    imname=annot.image.name # 获取第aid个图片的文件名
    all_imnames.append(imname) # 这里保存着注释文件中涉及的所有文件名，不会缺失（即从本地数据集中统计的可能会缺失）
    try:
      this_split=filename_to_split[imname[:-4]]
    except:
      continue # 注意：filename_to_split中的文件名是从本地读来的，极可能缺失，因此可能有的imname在里面索引不到，此时忽略该文件
  
    #-# 获取该图片的关节点数据，保存在points_fmted  (每个关节点的坐标 id is_visiable信息放在一个元组中， )
    points_fmted=[]
    if 'annorect' in dir(annot):  # 如果annotect是annot对象的属性
      all_rects=annot.annorect # 读取annorect对象
      #-#-# 将all_rects对象转化为其属性的数组（多此一举）
      if isinstance(all_rects, scipy.io.matlab.mio5_params.mat_struct):
        all_rects=np.array([all_rects])
      #-#-# 遍历all_rects下 每一个元素（其实是个属性对象）  但最后只要.annopoints.point下的东西
      for rect in all_rects:
        
        try:
          points=rect.annopoints.point
        except:
          continue
        if isinstance(points, scipy.io.matlab.mio5_params.mat_struct):  #!!!!!!!!!!!
          points=np.array([points])
        for point in points:
          #-#-#-# 读到is_visiable
          try:
            is_visiable=point.is_visiable if point.is_visiable in [1,0] else 0  
          except:
            is_visiable=0
          #-#-#-# 都封装进来
          points_rect.append(point.id,point.x,point.y,is_visiable) # 每个关节点的
        points_fmted.append(points_rect)
    [el.sort for el in points_fmted]
    
    # 将该图片的注释信息都封装进image_obj中，其中_get_action_class是读取动作类别的函数？？？？？？？？
    image_obj=(annot.image.name,
               _get_action_class
              points_fmted)
    
    _IMG_DIR = '/home/yzy/dataset/MPII/images/'
    if ios.path.exists(os.path.join(_IMG_DIR,imname)):
      lists_to_write[this_split].append(image_obj)
      img_id_in_split[this_split].append(aid+1)
  cls_ids=sorted(actclassname_to_id.items(),keys=operator.itemgetter(1))#？？？

  with  open(os.path.join(dataset_dir,'classes.txt'), 'w') as fout:
    fout.write('\n'.join([el[0]+';'+','.join(str(e) for e in list(el[1][1])) for el in cls_ids]))

  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDIrs(dataset_dir)

  # 只打乱训练集
  random.seed(_RANDOM_SEED)
  train_ids=range(len(lists_to_write['train']))
  random.shuffle(train_ids)
  lists_to_write['train']=[lists_to_write['train'][i] for i in train_ids]
  img_id_in_split['train']=[img_id_in_split['train'][i] for i in train_ids]

  with open(os.path.join(dataset_dir,'imnames.txt'),'w') as fout:
    fout.write('\n'.join(all_imnames))
  for spl in splits:
    with open(os.path.join(dataset_dir, '{}_ids.txt'.format(spl)), 'w') as fout:
      fout.write('\n'.join(str(el) for el in img_id_in_split[spl]))
    spl_name=spl


  # 最后转化tfrecord
  _convert_dataset(spl_name, lists_to_write[spl], dataset_dir)



if __name__=='__main__':
  main()