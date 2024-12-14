from model.generator import Generator
import torch
import utils.config
from utils import load_checkpoint


def get_gen():
    gen = Generator(in_channels=3, num_channels=64, num_blocks=16).to(DEVICE)
    load_checkpoint("gen.pth.tar", gen, None, None)
    gen.eval()
    return gen

def test():
    gen = get_gen()
    low_res = torch.randn((1, 3, 24, 24)).to(DEVICE)
    fake_high_res = gen(low_res)
    print(fake_high_res.shape)  # Expected output: torch.Size([1, 3, 96, 96])