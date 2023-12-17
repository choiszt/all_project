dependencies = ['torch', 'numpy', 'resampy', 'soundfile']

from torchvggish.vggish import VGGish
import torch

model_urls = {
    'vggish': '/home/zhouzhenyu/cond_adver/Verify/vggish_finetune/torchvggish-master/torchvggish/epoch2_loss711.pth',

}


def vggish(**kwargs):
    # print('实例化')
    model = VGGish(urls=model_urls, **kwargs)
    # for k ,v in model.state_dict().items():
    #     print(k)

    state_dict = torch.load('/home/zhouzhenyu/cond_adver/Verify/vggish_finetune/torchvggish-master/torchvggish/epoch2_loss711.pth')
    model.load_state_dict(state_dict)

    for name ,param in model.named_parameters():
        param.requires_grad = False
    

    for name ,param in model.named_parameters():
        if param.requires_grad:
            print('requires_grad: True',name)

    return model
