import torch
from model import U2NETP # small version u2net 4.7 MB

net = U2NETP(3,1)
net.load_state_dict(torch.load('./saved_models/u2netp/u2netp.pth'))
# if torch.cuda.is_available():
#     net.cuda()
net.eval()

torch_res = torch.onnx._export(net
                            , torch.rand(1, 3, 320, 320)
                            , 'assets/u2netp.onnx'
                            , export_params=True
                            , opset_version = 11)
