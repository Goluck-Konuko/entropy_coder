from entropy_models import GaussianConditional, EntropyBottleneck
import torch

class EntropyCoder():
    def __init__(self, entropy_bottleneck_channels, scale_table=None) -> None:
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)
        self.gaussian_conditional = GaussianConditional(scale_table=scale_table)



if __name__ == "__main__":
    M=10
    inp = torch.randn((1,10,4,4))
    print(inp[:,0,1:5,0])
    coder = EntropyCoder(M)
    # inp_hat, inp_likelihoods = coder.entropy_bottleneck(inp)
    inp_strings = coder.entropy_bottleneck.compress(inp)
    inp_hat = coder.entropy_bottleneck.decompress(inp_strings, size=(4,4))
    print(inp_hat[:,0,1:5,0])
    # print(inp_strings)
