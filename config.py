try:
    from configparser import ConfigParser
except:
    from ConfigParser import ConfigParser

from utils import AttrDict

import logging

logger = logging.getLogger(__name__)

CONFIG_SECS = [
    'train',
    'valid',
    'test',
    'infer',
]


def parse_config(cfg_file):
    parser = ConfigParser()
    cfg = AttrDict()
    parser.read(cfg_file)#读取配置文件
    for sec in parser.sections():#获得配置文件的所有区域.中括号“[ ]”内包含的为section。section 下面为类似于key-value 的配置内容
        sec_dict = AttrDict()
        for k, v in parser.items(sec):#获取属性和值
            try:
                v = eval(v)#执行一个字符串表达式，并返回表达式的值
            except:
                pass
            setattr(sec_dict, k, v)#用于设置属性值
        setattr(cfg, sec.upper(), sec_dict)

    return cfg


def merge_configs(cfg, sec, args_dict):
    assert sec in CONFIG_SECS, "invalid config section {}".format(sec)
    print(cfg)
    sec_dict = getattr(cfg, sec.upper())#返回一个对象属性值
    for k, v in args_dict.items():
        if v is None:
            continue
        try:
            if hasattr(sec_dict, k):
                setattr(sec_dict, k, v)
        except:
            pass
    return cfg


def print_configs(cfg, mode):
    logger.info("---------------- {:>5} Arguments ----------------".format(
        mode))
    for sec, sec_items in cfg.items():
        logger.info("{}:".format(sec))
        for k, v in sec_items.items():
            logger.info("    {}:{}".format(k, v))
    logger.info("-------------------------------------------------")
