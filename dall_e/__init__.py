import os
import wget
import paddle
from .model import Encoder, Decoder, EncoderBlock, DecoderBlock
from paddle.vision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

logit_laplace_eps = 0.1


def map_pixels(x):
    return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps


def unmap_pixels(x):
    return paddle.clip((x - logit_laplace_eps) / (1 - 2 * logit_laplace_eps), 0, 1)


model_dict = {
    'encoder': [Encoder, r'https://bj.bcebos.com/v1/ai-studio-online/bec111991e654908b6b6ebe3071a2c7d5bf0304af2064b089667715d81a0e746?responseContentDisposition=attachment%3B%20filename%3Dencoder.pdparams', 'encoder.pdparams'],
    'decoder': [Decoder, r'https://bj.bcebos.com/v1/ai-studio-online/9d0314834eb849c295a24ee75686c4a4339e669fe5dc41e78b34f564f6587c7a?responseContentDisposition=attachment%3B%20filename%3Ddecoder.pdparams', 'decoder.pdparams']
}


def load_model(model_name, pretrained=False):
    model_fn, url, file_name = model_dict[model_name]
    model = model_fn()

    if pretrained:
        model_path = os.path.join('pretrained_models', file_name)
        if not os.path.isfile(model_path):
            if not os.path.exists('pretrained_models'):
                os.mkdir('pretrained_models')
            wget.download(url, out=model_path)
        params = paddle.load(model_path)
        model.set_dict(params)

    model.eval()
    return model
