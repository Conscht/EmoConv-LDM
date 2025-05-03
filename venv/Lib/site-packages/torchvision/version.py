__version__ = '0.18.1+cpu'
git_version = '126fc22ce33e6c2426edcf9ed540810c178fe9ce'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
