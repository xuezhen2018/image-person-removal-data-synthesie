
from networks.resunet import SLBR as SLBR
from networks.resunetNM import SLBR as SLBRNM


# our method
def slbr(**kwargs):
    return SLBR(args=kwargs['args'], shared_depth=1, blocks=3, long_skip=True)

def slbrnm(**kwargs):
    return SLBRNM(args=kwargs['args'], shared_depth=1, blocks=3, long_skip=True)




