import paddle.fluid as fluid
import torch
import collections
from model.resnet_3d import ResNet_3d
def torch2paddle(torch_para, paddle_model,paddle_para_name=None):
    torch_state_dict = torch.load(torch_para)#加载模型参数

    paddel_state_dict = paddle_model.state_dict()
    #去掉bn中多余的参数
    tmp = []
    for key in torch_state_dict.keys():
        if ('_tracked' in key):
            tmp.append(key)
    for i in range(len(tmp)):
        torch_state_dict.pop(tmp[i])#删除给定键对应的值
    assert(len(torch_state_dict)==len(paddel_state_dict))#将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
    new_weight = collections.OrderedDict()#根据放入元素的先后顺序进行排序，所以输出的值是排好序的
    for torch_key,paddle_key in zip(torch_state_dict.keys(),paddel_state_dict.keys()):
        tmp = torch_state_dict[torch_key].detach().numpy()#detach()不参与参数更新
        if 'fc' in torch_key:
            new_weight[paddle_key]=tmp.T
        else:
            new_weight[paddle_key] = tmp
    paddle_model.set_dict(new_weight)
    if paddle_para_name==None:
        name = torch_para[0:-4]
        fluid.save_dygraph(paddle_model.state_dict(), name)
    else:
        fluid.save_dygraph(paddle_model.state_dict(),paddle_para_name)

if __name__ == "__main__":
    with fluid.dygraph.guard():
        paddle_model = ResNet_3d(50,400)
        torch2paddle('C:/Users/86135/Desktop/r3d50_KM_200ep.pth',paddle_model,'C:/Users/86135/Desktop/resnet_3d_model1')
