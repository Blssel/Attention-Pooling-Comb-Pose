#!/usr/env python
#coding:utf-8
import os

__author__='Zhiyu Yin'

_MPII_MAT_FILE='/home/yzy/dataset/MPII/mpii_human_pose_v1_u12_1.mat'

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
  '''
  annots=T['RELEASE'].annolist

  # 

if __name__=='__main__':
  main()