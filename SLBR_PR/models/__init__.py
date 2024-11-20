from .BasicModel import BasicModel
from .SLBR import SLBR  as SLBR
from .SLBRNM import SLBR as SLBRNM
from .FFPRN import FFPRN as FFPRN


def basic(**kwargs):
	return BasicModel(**kwargs)

def slbr(**kwargs):
    return SLBR(**kwargs)

def slbrnm(**kwargs):
    return SLBRNM(**kwargs)

def ffprn(**kwargs):
    return SLBRNM(**kwargs)
