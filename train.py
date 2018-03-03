import argparse
import os

# 获取所有需要训练的变量（保存在list中）
def _get_variables_to_train():
  if cfg.TRAIN.TRAINABLE_SCOPES == '':
    return tf.trainable_variables()
  else:
    # 将cfg中定义的所有需要涵盖可训练变量的scope都取出来
    scopes=[scope.strip() for scope in cfg.TRAIN.TRAINABLE_SCOPES.split(',')]
  # 获得这些scope下的变量
  variables_to_train=[]
  for scope in scopes:
    variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope)
    variables_to_train.append(variable)

  return variables_to_train
    
  

def _configure_optimizer(learning_rate):
  if cfg.TRAIN.OPTIMIZER == 'adam':

  # 如果是momentum的优化方式
  elif cfg.TRAIN.OPTIMIZER == 'momentum':
    optimizer=tf.train.MomentumOptimizer(learning_rate,
                                        momentum=cfg.TRAIN.MOMENTUM,
                                        name='Momentum')

  elif cfg.TRAIN.OPTIMIZER == 'rmsprop':

  elif cfg.TRAIN.OPTIMIZER == 'sgd':

  else:
    raise ValueError('Optimizer [%s] was not recognized',cfg.TRAIN.OPTIMIZER)

  return optimizer

def _configure_learning_rate(num_samples_per_epoch,num_clones,global_step):
  if cfg.NUM_STEPS_PER_DECAY>0:
    # 设置下降的步长
    decay_steps=cfg.NUM_STEPS_PER_DECAY
    tf.logging.info('Using {} steps for decay. Ignoring any epoch setting for '
                    'decay.'.format(decay_steps))
  else:
    # 如果没有在cfg中给出的话，则手动算！！！！为何这么算？？？？？？？？？
    decay_steps= int(num_samples_per_epoch / (cfg.TRAIN.BATCH_SIZE * num_clones * cfg.TRAIN.ITER_SIZE) * cfg.TRAIN.NUM_EPOCHS_PER_DECAY)
  
  # 选择衰减方式
  if cfg.TRAIN.LEARNING_RATE_DECAY_TYPE== 'exponential'
    return tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE,
                                      global_step,
                                      decay_steps,
                                      cfg.TRAIN.LEARNING_RATE_DECAY_RATE,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif cfg.TRAIN.LEARNING_RATE_DECAY_TYPE == 'fixed': #其它两种衰减方式

  elif cfg.TRAIN.LEARNING_RATE_DECAY_TYPE == 'polynomial':

  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',cfg.TRAIN.LEARNING_RATE_DECAY_RATE)


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

    # clone是对输出和名称nz空间的封装
    clones=model_deploy.creat_clones(deploy_config,clone_fn,[batch_queue])
    first_clone_scope=deploy_config.clone_scope(0)
    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS,first_clone_scope)
      

    from nets import nets_factory
    network_fn=net_factory.get_network_fn(cfg.MODEL_NAME,num_calsses=,num)#该函数作者又重新写过

    # 为每一个end_point节点加入监控
    for end_point in end_points:
      x= end_points[end_point]
      summaries.add(tf.summary.histogram('activations/'+ end_point, x))
    
    # 加入图片summary
    sum_img=tf.concat(tf.unstack(end_points['Image']))  # unstack作用是取消堆叠，也就是一帧一帧零散出来，用list包裹  concat感觉像是将所有图片按空间拼接起来，方便看每一帧
    if sum_img.get_shape().as_list()[-1] not in [1, 3, 4]:
      # 再做点处理  还不太懂？？？？？？？
    # 加入summary
    summaries.add(tf.summary.image('images',sum_img))
    
    # 加入由于加入pose而导致模型中新增的endpoi
    for epname in cfg.TRAIN.OTHER_IMG_SUMMARY_TO_ADD:    # OTHER_IMG_SUMMARIES_TO_ADD = ['PosePrelogitsBasedAttention']
      if epname in end_points:
        summary.add(tf.summary.image('image_vis/'+ epname, end_points[epname]))
    
    summaries=summaries.union()   # 求summaries和参数的并集，还赋给summaries？？？？？？？

    # 为loss增加summaries
    for loss in tf.get_collection(tf.Graphkeys.LOSSES,first_clone_scope):
      summaries.add(tf.summary.scalar(tensor=loss,name='losses/%s'% loss.op.name))
    
    # 为变量增加summies
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    # 配置滑动平均 (moving average)

    # 配置优化程序
    with tf.device(deploy_config.optimizer_device()):
      # 设置学习率
      learning_rate= _configure_learning_rate(dataset.num_samples, num_clones, global_step)
      # 
      optimizer=_configure_optimizer(learning_rate)
      summaries.add(tf.summary.scalar(tensor=learning_rate,name='learning_rate'))

    # 设置哪些变量需要参与训练
    variables_to_train=_get_variables_to_train()
    








