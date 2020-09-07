import os
import sys
import time
import logging
import argparse
import ast
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import paddle.fluid as fluid

from model.resnet_3d import ResNet_3d
from ucf101_reader import Ucf101Reader
from config import parse_config, merge_configs, print_configs

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        default='c3d',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='model/c3d_ucf101.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--weights',
        type=str,
        default='./model/check/resnet_3d_model.pdparams',
        help='weight path, None to use weights from Paddle.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='sample number in a batch for inference.')

    args = parser.parse_args()
    return args


def eval(args):
    # parse config
    config = parse_config(args.config)
    val_config = merge_configs(config, 'test', vars(args))
    print_configs(val_config, "test")
    with fluid.dygraph.guard():
        val_model = ResNet_3d()
       # get infer reader
        test_reader = Ucf101Reader(args.model_name.upper(), 'test', val_config).create_reader()

        # if no weight files specified, exit()
        if args.weights:
            weights = args.weights
        else:
            print("model path must be specified")
            exit()
            
        para_state_dict, _ = fluid.load_dygraph(weights)
        val_model.load_dict(para_state_dict)
        val_model.eval()
        
        acc_list = []
        acc_list_local=[]
        for batch_id, data in enumerate(test_reader()):
           
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([[x[1]] for x in data]).astype('int64')
            
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)
            label.stop_gradient = True
            
            out = val_model(img)
            acc = fluid.layers.accuracy(out,label)
            acc_list.append(acc.numpy()[0])
            acc_list_local.append(acc.numpy()[0])
            if((batch_id % 300) == 0 and(batch_id!=0)):
                logger.info("valid Loss at step {}:  acc: {}".format(batch_id, np.mean(acc_list_local)))
                print("valid Loss at  step {}:  acc: {}".format(batch_id,  np.mean(acc_list_local)))
                acc_list_local=[]

        print("验证集准确率为:{}".format(np.mean(acc_list)))

            
            
            
if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    eval(args)
