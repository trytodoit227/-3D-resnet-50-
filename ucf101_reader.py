import os
import sys
import cv2
import math
import random
import functools
import glob
try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO
import numpy as np
import paddle
import paddle.fluid as fluid


from PIL import Image, ImageEnhance
import logging



logger = logging.getLogger(__name__)
python_ver = sys.version_info


class Ucf101Reader(object):
    """
    Data reader for kinetics dataset of two format mp4 and pkl.
    1. mp4, the original format of kinetics400
    2. pkl, the mp4 was decoded previously and stored as pkl
    In both case, load the data, and then get the frame data in the form of numpy and label as an integer.
     dataset cfg: format
                  num_classes
                  seg_num
                  short_size
                  target_size
                  num_reader_threads
                  buf_size
                  image_mean
                  image_std
                  batch_size
                  list
    """

    def __init__(self, name, mode, cfg):
        self.cfg = cfg
        self.mode = mode
        self.name = name
        self.format = cfg.MODEL.format
        self.num_classes = self.get_config_from_sec('model', 'num_classes')
        if(mode != 'TEST'):
            self.seg_num = self.get_config_from_sec('model', 'seg_num')
        else:
            self.seg_num = self.get_config_from_sec(mode, 'seg_num', self.seg_num)
        self.seglen = self.get_config_from_sec('model', 'seglen')
        self.short_size = self.get_config_from_sec(mode, 'short_size')
        self.target_size = self.get_config_from_sec(mode, 'target_size')
        self.num_reader_threads = self.get_config_from_sec(mode,
                                                           'num_reader_threads')
        self.buf_size = self.get_config_from_sec(mode, 'buf_size')
        self.fix_random_seed = self.get_config_from_sec(mode, 'fix_random_seed')

        self.img_mean = np.array(cfg.MODEL.image_mean).reshape(
            [3, 1, 1]).astype(np.float32)
        self.img_std = np.array(cfg.MODEL.image_std).reshape(
            [3, 1, 1]).astype(np.float32)
        # set batch size and file list
        self.batch_size = cfg[mode.upper()]['batch_size']
        self.filelist = cfg[mode.upper()]['filelist']
        # set num_trainers and trainer_id when distributed training is implemented
        self.num_trainers = self.get_config_from_sec(mode, 'num_trainers', 1)
        self.trainer_id = self.get_config_from_sec(mode, 'trainer_id', 0)


     
        self.video_path = cfg[mode.upper()]['video_path']
        def create_classInt(class_num,path):
            assert os.path.exists(path),\
                 '{} not exist, please check the classInd file'.format(path)
            classInd = dict()
            with open(path) as flist:  
                for line in flist:
                    temp = line.split()#以空格为分隔符
                    classInd[temp[1]] = int(temp[0])#将每一行的第一个元素作为第二个元素的值
            assert len(classInd) ==  class_num, \
                  'Num of class is wrong'
            return classInd #构建classlnd
       
        self.classInd = create_classInt(self.num_classes,
                        self.get_config_from_sec('model', 'classind'))
        
        if self.fix_random_seed:
            random.seed(0)
            np.random.seed(0)
            self.num_reader_threads = 1

    def get_config_from_sec(self, sec, item, default=None):
        if sec.upper() not in self.cfg:#配置cfg文件
            return default
        return self.cfg[sec.upper()].get(item, default)

    def create_reader(self):

       # if set video_path for inference mode, just load this single video
        if (self.mode == 'infer') and (self.video_path != ''):
            # load video from file stored at video_path
            print('No infer mode')

        else:
            assert os.path.exists(self.filelist), \
                        '{} not exist, please check the data list'.format(self.filelist)
            _reader = self._reader_creator(pickle_list=self.filelist, mode = self.mode, seg_num=self.seg_num, seglen = self.seglen, \
                             short_size = self.short_size, target_size = self.target_size, \
                             img_mean = self.img_mean, img_std = self.img_std, \
                             shuffle = (self.mode == 'train'), \
                             num_threads = self.num_reader_threads, \
                             buf_size = self.buf_size, format = self.format)

        def _batch_reader():
            batch_out = []
            for imgs, label in _reader():
                if imgs is None:
                    continue
                batch_out.append((imgs, label))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []

        return _batch_reader

    def _reader_creator(self,
                        pickle_list,
                        mode,
                        seg_num,
                        seglen,
                        short_size,
                        target_size,
                        img_mean,
                        img_std,
                        shuffle=False,
                        num_threads=1,
                        buf_size=1024,
                        format='jpg'):
       

        def decode_pickle(sample, mode, seg_num, seglen, short_size,
                          target_size, img_mean, img_std):
            pickle_path = sample[0]
            try:
                if python_ver < (3, 0):
                    data_loaded = pickle.load(open(pickle_path, 'rb'))
                else:
                    data_loaded = pickle.load(
                        open(pickle_path, 'rb'), encoding='bytes')

                vid, label, frames = data_loaded
                if len(frames) < 1:
                    logger.error('{} frame length {} less than 1.'.format(
                        pickle_path, len(frames)))
                    return None, None
            except:
                logger.info('Error when loading {}'.format(pickle_path))
                return None, None

            if mode == 'train' or mode == 'valid' or mode == 'test':
                ret_label = label
            elif mode == 'infer':
                ret_label = vid

            imgs = video_loader(frames, seg_num, seglen, mode)
            return imgs_transform(imgs, mode, seg_num, seglen, \
                         short_size, target_size, img_mean, img_std, name = self.name), ret_label
       
        def decode_jpg(sample,classInd,vid_path, mode, seg_num, seglen, short_size,\
                          target_size, img_mean, img_std):
            pickle_path = sample[0].split('/')[1]#用/将其分开
            pickle_path = pickle_path.split('.')[0]#用.将其分开
            ret_label = classInd[sample[0].split('/')[0]]-1
            jpg_path = os.path.join( vid_path,pickle_path)#生成图片的路径
            #print(sample)
            #print(jpg_path)
            if(not(os.path.exists(jpg_path))):
                logger.error('{} path not exists'.format(jpg_path))
                return None, None
         #读入原始尺寸图片 
            imgs= video_loader(jpg_path, seg_num, seglen, mode)
            return imgs_transform(imgs, mode, seg_num, seglen, \
                         short_size, target_size, img_mean, img_std, name = self.name), ret_label

        def reader_():
            with open(pickle_list) as flist:
                full_lines = [line.split(' ') for line in flist]
                if self.mode == 'train':
                    if (not hasattr(reader_, 'seed')):
                        reader_.seed = 0
                    random.Random(reader_.seed).shuffle(full_lines)
                    print("reader shuffle seed", reader_.seed)
                    if reader_.seed is not None:
                        reader_.seed += 1  
                per_node_lines = int(
                    math.ceil(len(full_lines) * 1.0 / self.num_trainers))
                total_lines = per_node_lines * self.num_trainers

                # aligned full_lines so that it can evenly divisible
                full_lines += full_lines[:(total_lines - len(full_lines))]
                assert len(full_lines) == total_lines

                # trainer get own sample
                lines = full_lines[self.trainer_id:total_lines:
                                   self.num_trainers]
                logger.info("trainerid %d, trainer_count %d" %
                            (self.trainer_id, self.num_trainers))
                logger.info(
                    "read images from %d, length: %d, lines length: %d, total: %d"
                    % (self.trainer_id * per_node_lines, per_node_lines,
                       len(lines), len(full_lines)))
                assert len(lines) == per_node_lines
                for line in full_lines:
                    pickle_path = line[0]
                    yield [pickle_path]

        if format == 'pkl':
            decode_func = decode_pickle
        elif format == 'jpg':
            decode_func = decode_jpg
        else:
            raise "Not implemented format {}".format(format)

        mapper = functools.partial(
            decode_func,
            classInd=self.classInd,
            vid_path=self.video_path,
            mode=mode,
            seg_num=seg_num,
            seglen=seglen,
            short_size=short_size,
            target_size=target_size,
            img_mean=img_mean,
            img_std=img_std)
        return fluid.io.xmap_readers(mapper, reader_, num_threads, buf_size)
#        return fluid.io.map_readers(mapper, reader_)


def imgs_transform(imgs,
                   mode,
                   seg_num,
                   seglen,
                   short_size,
                   target_size,
                   img_mean,
                   img_std,
                   name=''):
    imgs = group_scale(imgs, short_size)#为什么这里用short_size
    #imgs = group_corner_crop(imgs,target_size,not(mode == 'train'))
    if mode == 'train':
        imgs = group_random_flip(imgs)
        #imgs = group_multi_scale_crop(imgs, short_size)
        imgs = group_random_crop(imgs,target_size)
        #imgs=addnoise(imgs)
        #imgs = group_corner_crop(imgs, target_size,fix_center=False)
    else:
        imgs = group_center_crop(imgs,target_size)
        

#     if mode == 'train':
# #        imgs = group_multi_scale_crop(imgs, short_size)
#         imgs = group_random_crop(imgs, target_size)
#         imgs = group_random_flip(imgs)
#     else:
#         imgs = group_center_crop(imgs, target_size)
#读入图像是h×w×c，调整为c×h×w
    np_imgs = (np.array(imgs[0]).astype('float32').transpose(
        (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
    for i in range(len(imgs) - 1):
        img = (np.array(imgs[i + 1]).astype('float32').transpose(
            (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
        np_imgs = np.concatenate((np_imgs, img))
    imgs = np_imgs
    imgs -= img_mean
    imgs /= img_std
    imgs = np.reshape(imgs, (-1, seglen * 3, target_size, target_size))

    return imgs

def group_corner_crop(img_group,target_size,fix_center=False):
    #计算采样窗口的大小和左上角坐标
    def corner_crop(img,fix_center=False):
        #定义尺度和采样位置
        scales = [1,0.85,0.7,0.6,0.5]
        crop_positions = ['c', 'tl', 'tr', 'bl', 'br']
        #获得随机的尺度和采样位置
        if(fix_center==False):
            crop_position = crop_positions[random.randint(0, len(crop_positions) - 1)]#随机选择一个位置
            crop_scale = scales[random.randint(0, len(scales) - 1)]#生成尺度随机数
        else:
            crop_position = crop_positions[0]
            crop_scale = scales[0]
        im_size = img.size
        crop_size = min(im_size[0],im_size[1])
        crop_size = int(crop_size*crop_scale)
        image_width =im_size[0]
        image_height = im_size[1]

        h, w = (crop_size, crop_size)
        if crop_position == 'c':
            i = int(round((image_height - h) / 2.))
            j = int(round((image_width - w) / 2.))
        elif crop_position == 'tl':
            i = 0
            j = 0
        elif crop_position == 'tr':
            i = 0
            j = image_width - w
        elif crop_position == 'bl':
            i = image_height - h
            j = 0
        elif crop_position == 'br':
            i = image_height - h
            j = image_width - w
        return crop_size,crop_size,i,j

    crop_w, crop_h, offset_w, offset_h = corner_crop(img_group[0],fix_center)
    crop_img_group = [
        img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))#裁剪左上右下指定尺寸的图片
        for img in img_group
    ]
    ret_img_group = [
        img.resize((target_size, target_size), Image.BILINEAR)#改变图像大小
        for img in crop_img_group
    ]
    return ret_img_group



def group_multi_scale_crop(img_group, target_size, scales=None, \
        max_distort=1, fix_crop=True, more_fix_crop=True):
    scales = scales if scales is not None else [1, .875, .75, .66]
    input_size = [target_size, target_size]

    im_size = img_group[0].size

    # get random crop offset
    def _sample_crop_size(im_size):
        image_w, image_h = im_size[0], im_size[1]

        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in scales]
        crop_h = [
            input_size[1] if abs(x - input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            input_size[0] if abs(x - input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_step = (image_w - crop_pair[0]) / 4
            h_step = (image_h - crop_pair[1]) / 4

            ret = list()
            ret.append((0, 0))  # upper left
            if w_step != 0:
                ret.append((4 * w_step, 0))  # upper right
            if h_step != 0:
                ret.append((0, 4 * h_step))  # lower left
            if h_step != 0 and w_step != 0:
                ret.append((4 * w_step, 4 * h_step))  # lower right
            if h_step != 0 or w_step != 0:
                ret.append((2 * w_step, 2 * h_step))  # center

            if more_fix_crop:
                ret.append((0, 2 * h_step))  # center left
                ret.append((4 * w_step, 2 * h_step))  # center right
                ret.append((2 * w_step, 4 * h_step))  # lower center
                ret.append((2 * w_step, 0 * h_step))  # upper center

                ret.append((1 * w_step, 1 * h_step))  # upper left quarter
                ret.append((3 * w_step, 1 * h_step))  # upper right quarter
                ret.append((1 * w_step, 3 * h_step))  # lower left quarter
                ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

            w_offset, h_offset = random.choice(ret)

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    crop_w, crop_h, offset_w, offset_h = _sample_crop_size(im_size)
    crop_img_group = [
        img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        for img in img_group
    ]
    ret_img_group = [
        img.resize((input_size[0], input_size[1]), Image.BILINEAR)
        for img in crop_img_group
    ]

    return ret_img_group


def group_random_crop(img_group, target_size):
    w, h = img_group[0].size
    th, tw = target_size, target_size

    assert (w >= target_size) and (h >= target_size), \
          "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)

    out_images = []
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    for img in img_group:
        if w == tw and h == th:
            out_images.append(img)
        else:
            out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return out_images


def group_random_flip(img_group):
    v = random.random()#用于生成一个0到1的随机符点数
    if v < 0.5:
        ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]#水平翻转
        return ret
    else:
        return img_group


def group_center_crop(img_group, target_size):
    img_crop = []
    for img in img_group:
        w, h = img.size
        th, tw = target_size, target_size
        assert (w >= target_size) and (h >= target_size), \
             "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)
        x1 = int(round((w - tw) / 2.))#返回浮点数x的四舍五入值
        y1 = int(round((h - th) / 2.))
        img_crop.append(img.crop((x1, y1, x1 + tw, y1 + th)))#截取tw,th的图像

    return img_crop


def group_scale(imgs, target_size):#调整图片的尺寸大小
    resized_imgs = []
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.size#图片的宽和高
        if (w <= h and w == target_size) or (h <= w and h == target_size):
            resized_imgs.append(img)
            continue

        if w < h:
            ow = target_size
            oh = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
        else:
            oh = target_size
            ow = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))#图片尺寸的改变，双线性

    return resized_imgs

#增加噪声
def addnoise(imgs):
    imgggs=[]
    for i in range(len(imgs)):
        img = imgs[i]
        #print('img.size'+img.size)
        w, h = img.size#图片的宽和高
        #print('w',w)
        #print('h',h)
        imgg=np.array(img)
        #print(imgg.shape)
        noise=np.random.randint(5, size=(w,h,3),dtype='uint8')
        imgggs.append(Image.fromarray(imgg+noise))
    return imgggs


def imageloader(buf):
    if isinstance(buf, str):
        img = Image.open(StringIO(buf))
    else:
        img = Image.open(BytesIO(buf))

    return img.convert('RGB')


def video_loader(path, nsample, seglen, mode):###这里的path有问题
    frames = glob.glob(os.path.join(path,'*.jpg'))#获取指定目录下的全部图片
    videolen = len(frames)#一共有多少图片
    if(mode != 'test'):
        seg_num = nsample#一段视频划分为几段
    else:
        seg_num = videolen//seglen
        seg_num = max(nsample,seg_num)
    # seg_num = nsample
    
    average_dur = int(videolen / seg_num)#平均每段有多少张图片

    imgs = []
    for i in range(seg_num):#遍历每一段
        idx = 0
        if mode == 'train':
            #训练时当每段帧数大于采样数时，每段开始采样的位置是随机选取的
            if average_dur >= seglen:
                idx = random.randint(0, average_dur - seglen)#生成指定范围内的随机整数
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i
        else:
            #测试时当每段帧数大于采样数时，每段开始采样的位置固定为每段的中点
            if average_dur >= seglen:
                idx = (average_dur - seglen) // 2
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i

        for jj in range(idx, idx + seglen):
 #           index = frames[int(jj % videolen)]
            index = int(jj % videolen)
            file_name = os.path.join(path,'frame{:06d}.jpg'.format(index+1))
            img = Image.open(file_name)#加载该路径下的图片
            imgs.append(img)

    return imgs
