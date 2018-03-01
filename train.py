import argparse
import os

def parse_args():
  parser=argparse.ArgumentParser(description='Train a keypoint regressor.')
  parser.add('--cfg',dest='cfg_file',help='optional config file',default=None,type=str)
  args=parser.parse_args()
  return args
def main():
  args=parse_args()
  # 将args.cfg_file融合到cfg中
  from config import cfg, cfg_from_file
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)

  tf.logging.info('Using Config:')
  pprint.pprint(cfg)
  
  # 指定或创建文件夹?????
  from config import get_output_dir
  train_dir = get_output_dir('default' if args.cfg_file is None else args.cfg_file)

  # 指定使用哪块GPU
  os.environ['CUDA_VISIBLE_DEVICES']=cfg.GPUS
  num_clones=len(cfg.GPUS.split(','))

  # 设置日志显示级别
  tf.logging.set_verbosity(tf.logging.INFO)

  # 创建计算图，并设置为默认图
  with tf.Graph().as_default():
    tf.set_random_seed(cfg.RNG_SEED)#？？？？？
    # 关于之后要单间的模型的部署设置
    deploy_config=model_deploy.DeploymentConfig(num_clones=num_clones,clone_on_cpu=False,replica_id=0,num_replica=1,num_ps.task=0)
    
    # 创建global_step
    with tf.device(deploy_config.variables_device()):
      global_step=slim.creat_global_step()

    # ------------------------------数据集------------------------------#
    kwargs={}# 保存关于如何使用视频的超参数视频
    if cfg.TRAIN.VIDEO_FRAMES_PER_VIDEO>1:
      kwargs['num_samples']=cfg.TRAIN.VIDEO_FRAMES_PER_VIDEO
      kwargs['randomFromSegmentStyle']=cfg.TRAIN.READ_SEGMENT_STYLE
      kwargs['modality'] = cfg.INPUT.VIDEO.MODALITY   #输入模态：默认为rgb
      kwargs['split_id'] = cfg.INPUT.SPLIT_ID
      # 还有俩不知啥意思？？？？？？

    # 选择预处理函数(也是作者重新修改过的！！！！！！！！！！！！！！！)
    from preprocessing import preprocessing_factory
    image_preprocessing_fn=preprocessing_factory.get_preprocessing(preprocessing_name,is_training=True)
    
    # 读取数据——获取Dataset对象
    from datasets import dataset_factory #注意此datasets在本目录下，是作者自己编写的,其中的get_dataset函数没有发生变化,只是其调用的函数选项发生变化，是作者自定义的，返回Dataset对象和一个整数
    dataset,num_pose_keypoints=dataset_factory.get_dataset(cfg.DATASET_NAME,cfg.DATASET_SPLIT_NAME,cfg.DATASET_DIR,**kwargs)
    # 读取数据——创建provider，读取+预处理，打包成batch，建立预取队列！！！！！
    with tf.device(deploy_config.inputs_device()):
      provider=slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                              num_readers= cfg.NUM_READERS,
                                                              common_queue_capacity= 20*cfg.TRAIN.BATCH_SIZE,
                                                              common_queue_min= 10*cfg.TRAIN.BATCH_SIZE)
      from preprocess_pipeline import train_preprocess_pipeline #该函数依据provider 和image_preprocessing_cn作为参数，从provider中读数据并且预处理
      [image,pose_label_hmap,pose_label_valid,action_label]=train_preprocess_pipeline(provider,cfg,     ,image_preprocessing_fn)# 真正读取数据？？？？？？？？？？？？？？
      # 打包batch
      images,pose_labels_hmap,pose_labels_valid,action_labels=tf.train.batch([image,pose_label_hmap,pose_label_valid,action_label],
                                                                              batch_size=cfg.TRAIN.BATCH_SIZE,
                                                                              num_thread=cfg.NUM_PREPROCESSING_THREADS,
                                                                              capacity=5*cfg.TRAIN.BATCH_SIZE)
      # 建立数据读取队列
      batch_queue=slim.prefetch_queue.prefetch_queue([images,pose_labels_hmap,pose_labels_valid,action_labels],
                                                      capacity=5*deploy_config.clones.cfg.TRAIN.ITER_SIZE)
      
    # ------------------------------选择网络?????------------------------------#
    def clone_fn(batch_queue):
      # 出队一个batch
      images,labels_pose,labels_pose_valid,labels_action=batch_queue.dequeue()
      labels_pose=tf.concat(tf.unstack(labels_pose),axis=0)
      labels_pose_valid=tf.concat(tf.unstack(labels_pose_valid),axis=0)
      
      # 前传(输入images)  注意:网络不仅会输出分类logits，还会输出姿态，但姿态输出记录在end_points
      logits,end_points=network_fn(images)
      pose_logits=end_points['PoseLogits']
      
      # 指定loss function 并计算loss
      # 该作者把一切都存进end_points里面了，
      end_points['Images']= images           # 存储信息只end_points中
      end_points['PoseLabels']= labels_pose
      end_points['ActionLabels']= labels_action
      end_points['ActionLogits']=logits
      gen_loss(labels_action,logits,cfg.TRAIN.LOSS_FN_ACTION,
              dataset.num_calsses,cfg.TRAIN.LOSS_FN_ACTION_WT,
              labels_pose,pose_logits,cfg.TRAIN.LOSS_FN_POSE,
              labels_pose_valid,cfg.TRAIN.LOSS_FN_POSE_WT,
              end_points,cfg)# 计算loss 该函数在loss模块中，loss.py就在当前路径下??????????????????

      return end_points
      
    # 收集summary
    summaries=set(tf.get_collection(tf.GRAPH.SUMMARIES))

    # clone是对输出和名称空间的封装
    clones=model_deploy.creat_clones(deploy_config,clone_fn,[batch_queue])
    first_clone_scope=deploy_config.clone_scope(0)
    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS,first_clone_scope)
      

    from nets import nets_factory
    network_fn=net_factory.get_network_fn(cfg.MODEL_NAME,num_calsses=,num)#该函数作者又重新写过



































    
  
  