import datetime
import logging
import shutil
import os
from pathlib import Path

def log(args,rank,split='train'):

    '''CREATE DIR'''
    # 获取当前时间的时间戳，并将其转化为字符串形式，以便作为实验记录目录名的一部分
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    # 创建一个名为"log"的目录，并确保该目录存在，如果不存在则创建该目录；
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    # 网络模型的名字
    exp_dir = exp_dir.joinpath(args.model_name)
    exp_dir.mkdir(exist_ok=True)
    # if args.log_dir is None:
    exp_dir = exp_dir.joinpath(timestr+args.add_label)
    # else:
    #     exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    # 存储训练的代码
    file_train=exp_dir.joinpath('file_train')
    file_train.mkdir(exist_ok=True)
    # 存储训练图像
    # train_pic = exp_dir.joinpath('train_pic')
    # train_pic.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")  # 创建一个名为Model的日志记录器logger
    logger.setLevel(logging.INFO)  # 设置了日志记录器logger的日志级别为IFO,即只记录IBFO级别以上的日志信息
    # 日志格式化器 设置日志记录的格式。 时间-记录器名称-日志级别-内容
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 用于将日志写入文件
    file_handler = logging.FileHandler('%s/%s.txt' % (exp_dir, split+"_logs"))
    # 日志级别为 INFO，即只记录级别为 INFO 及以上的日志消息。
    file_handler.setLevel(logging.INFO)
    # 将指定格式的日志文件写入到指定的日志文件中
    file_handler.setFormatter(formatter)
    # 这行代码将之前创建的file_handler对象添加到logger对象中
    logger.addHandler(file_handler)
    if rank ==0:
        logger.info('PARAMETER ...')
        logger.info('Lamda {}, add_label {}, model_name {} dataset_name {} '.format
                    (args.Lamda,args.add_label,args.model_name,args.dataset_name))
        logger.info('other information {}  '.format(args.other_info))
    return exp_dir,logger,timestr,file_train

def save_model_file(args,file_train):

    source_directory = f'./models/{args.model_name}'

    for root, dirs, files in os.walk(source_directory):
        for file in files:
            if file.endswith(".py"):  # 检查文件扩展名是否为 .py
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(file_train, file)

                # 复制文件到目标目录
                shutil.copy(source_file_path, target_file_path)
    # for root, dirs, files in os.walk(util_files ):
    #     for file in files:
    #         if file.endswith(".py"):  # 检查文件扩展名是否为 .py
    #             source_file_path = os.path.join(root, file)
    #             target_file_path = os.path.join(file_train, file)
    #
    #             # 复制文件到目标目录
    #             shutil.copy(source_file_path, target_file_path)
    #
    # for root, dirs, files in os.walk(main_files ):
    #     for file in files:
    #         if file.endswith(".py"):  # 检查文件扩展名是否为 .py
    #             source_file_path = os.path.join(root, file)
    #             target_file_path = os.path.join(file_train, file)
    #
    #             # 复制文件到目标目录
    #             shutil.copy(source_file_path, target_file_path)