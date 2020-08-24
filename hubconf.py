dependencies = ['torch', 'torchvision']

import torch


class _Normalize(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=-1)


def imagenet_resnet50_encoder(*, with_linear_clf=False):
    r"""
    ResNet50 encoder trained on full ImageNet using L_align and L_uniform
    losses.

    This checkpoint achieves 67.694% validation linear classification top1
    accuracy.

    Arguments:
        with_linear_clf (bool, optional, keyword-only): If True, returns an
        encoder with the last fully-connected layer being the trained linear
        classifier. If False, returns an encoder with the original MLP head,
        outputing an l2-normalized 128-dimensional vector. Default: False.
    """
    import torchvision.models

    if with_linear_clf:
        model = torchvision.models.resnet50(pretrained=False, num_classes=1000)

        ckpt = torch.hub.load_state_dict_from_url('https://github.com/SsnL/moco_align_uniform/releases/download/v1.0-checkpoints/imagenet_align_uniform_with_linear_clf.pth.tar')

        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            assert k.startswith('module.')
            state_dict[k.split('.', 1)[1]] = v
    else:
        model = torchvision.models.resnet50(pretrained=False, num_classes=128)
        dim_mlp = model.fc.weight.shape[1]
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(dim_mlp, dim_mlp),
            torch.nn.ReLU(),
            model.fc,
            _Normalize(),
        )

        ckpt = torch.hub.load_state_dict_from_url('https://github.com/SsnL/moco_align_uniform/releases/download/v1.0-checkpoints/imagenet_align_uniform.pth.tar')

        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            if k.startswith('module.encoder_q.'):
                state_dict[k.split('.', 2)[-1]] = v

    model.load_state_dict(state_dict)
    return model

