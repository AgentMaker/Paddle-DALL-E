# Paddle-DALL-E
A PaddlePaddle version implementation of DALL-E of OpenAI. [【origin repo】](https://github.com/openai/DALL-E)

Now this implementation only include the dVAE part, can't generate images from text.

## Quick Start
```python
import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T
import paddle.vision.transforms.functional as TF

from PIL import Image
from dall_e import load_model, map_pixels, unmap_pixels

target_image_size = 256

def preprocess(img):
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation='lanczos')
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = paddle.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)

enc = load_model('encoder', pretrained=True)
dec = load_model('decoder', pretrained=True)

img = Image.open('1000x-1.jpg')
x = preprocess(img)

z_logits = enc(x)
z = paddle.argmax(z_logits, axis=1)
z = F.one_hot(z, num_classes=enc.vocab_size).transpose((0, 3, 1, 2))

x_stats = dec(z)
x_rec = unmap_pixels(F.sigmoid(x_stats[:, :3]))

out = (x_rec[0].transpose((1, 2, 0))*255.).astype('uint8').numpy()
out = Image.fromarray(out)
out.show()
```