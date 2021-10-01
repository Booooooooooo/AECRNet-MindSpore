from mindspore import export, load_checkpoint, load_param_into_net
from mindspore import Tensor, context
import numpy as np
from models.model import Dehaze

#context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
net = Dehaze(3, 3)
param_dict = load_checkpoint("./trained_models/AECRNet_rdin_87.ckpt")

load_param_into_net(net, param_dict)
input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
export(net, Tensor(input), file_name='AECRNet_rdin_87', file_format='MINDIR')