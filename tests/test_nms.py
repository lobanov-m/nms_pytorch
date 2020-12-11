import torch

from nms_pytorch import nms, soft_nms


def test_nms():
    boxes = torch.tensor([[0, 0, 100, 100, 0.99], [0, 0, 101, 101, 0.95]])
    out = nms(boxes, 0.5)
    print(out)


if __name__ == '__main__':
    test_nms()