import os
import sys
import time
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid
from collections import OrderedDict

from model.resnet_3d import ResNet_3d
from ucf101_reader import Ucf101Reader
from config import parse_config, merge_configs, print_configs

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(filename='logger.log', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")#创建 ArgumentParser() 对象
    parser.add_argument(
        '--model_name',
        type=str,
        default='c3d',
        help='name of model to train.')#添加参数
    parser.add_argument(
        '--config',
        type=str,
        default='./model/c3d_ucf101.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default='./data/data50130',
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--resume',
        type=ast.literal_eval,
        default=True,
        help=''
    )
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./model/check',
        help='directory name to save train snapshoot')
    args = parser.parse_args()#解析参数
    return args


def train(args):
    # parse config
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        config = parse_config(args.config)
        train_config = merge_configs(config, 'train', vars(args))
        valid_config = merge_configs(config, 'valid', vars(args))
        print_configs(train_config, 'train')

        #根据自己定义的网络，声明train_model
        train_model=ResNet_3d()
        train_model.train()
        opt = fluid.optimizer.Momentum(config.TRAIN.learning_rate, 0.9, parameter_list=train_model.parameters(),
        regularization=fluid.regularizer.L2Decay(config.TRAIN.l2_weight_decay))
        
        #加载预训练参数
        #加载上一次训练好的模型
        if args.resume==True:
            model, _ = fluid.dygraph.load_dygraph(args.save_dir + '/resnet_3d_model.pdparams')
            train_model.load_dict(model)
            print('Resueme from '+ args.save_dir + '/resnet_3d_model.pdparams')
        # elif args.pretrain:
        #     pretrain_weights = fluid.io.load_program_state(args.pretrain)
        #     inner_state_dict = train_model.state_dict()
        #     print('Pretrain with '+ args.pretrain)
        #     for name, para in inner_state_dict.items():
        #         if((para.name in pretrain_weights) and (not('fc' in para.name))):
        #             para.set_value(pretrain_weights[para.name])
        #         else:
        #             print('del '+ para.name)
        #用3D参数初始化
        elif args.pretrain:
            pretrain_weights=fluid.io.load_program_state(args.pretrain+'/resnet_3d_model1.pdparams')#预训练模型转为之后的参数
            #print(a)
            inner_state_dict = train_model.state_dict()
            print('pretrain with'+args.pretrain)
            for name,para in inner_state_dict.items():
                if ((name in pretrain_weights) and (not('fc' in para.name))):
                    para.set_value(pretrain_weights[name])
                else:
                    print('del'+para.name)
            #train_model.set_dict(a)
        else:
            pass;
    


        # build model
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # get reader
        train_config.TRAIN.batch_size = train_config.TRAIN.batch_size
        train_reader = Ucf101Reader(args.model_name.upper(), 'train', train_config).create_reader()
        valid_reader = Ucf101Reader(args.model_name.upper(), 'valid', valid_config).create_reader()
        epochs = args.epoch or train_config.TRAIN.epoch
        #print(epochs)
        for i in range(epochs):
            train_model.train()#启用 BatchNormalization 和 Dropout
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([[x[1]] for x in data]).astype('int64')
                
                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True
                
#                out, acc = train_model(img, label)
                #print(img.shape)
                out = train_model(img)
                acc = fluid.layers.accuracy(out,label)
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()

                opt.minimize(avg_loss)
                train_model.clear_gradients()
                
                
                if batch_id % 10 == 0:
                    logger.info("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
                    print("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
            fluid.dygraph.save_dygraph(train_model.state_dict(), args.save_dir + '/resnet_3d_model')
            

            if((i%3)==0 and i!=0):
                acc_list = []
                avg_loss_list = []
                train_model.eval()
                for batch_id, data in enumerate(valid_reader()):
                    dy_x_data = np.array([x[0] for x in data]).astype('float32')
                    y_data = np.array([[x[1]] for x in data]).astype('int64')

                    img = fluid.dygraph.to_variable(dy_x_data)
                    label = fluid.dygraph.to_variable(y_data)
                    label.stop_gradient = True
                    out = train_model(img)
                    acc = fluid.layers.accuracy(out, label)
                    loss = fluid.layers.cross_entropy(out, label)
                    avg_loss = fluid.layers.mean(loss)
                    acc_list.append(acc.numpy()[0])
                    avg_loss_list.append(avg_loss.numpy())
                    if batch_id %20 == 0:
                        logger.info("valid Loss at step {}: {}, acc: {}".format(batch_id, avg_loss.numpy(), acc.numpy()))
                        print("valid Loss at  step {}: {}, acc: {}".format(batch_id, avg_loss.numpy(), acc.numpy()))
                print("验证集准确率为:{}".format(np.mean(acc_list)))
                print("验证集loss为:{}".format(np.mean(avg_loss_list)))

        #
        # logger.info("Final loss: {}".format(avg_loss.numpy()))
        # print("Final loss: {}".format(avg_loss.numpy()))

                


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    train(args)
