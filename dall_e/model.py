import paddle
import paddle.nn as nn


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class EncoderBlock(nn.Layer):
    def __init__(self, n_in, n_out, n_layers):
        super(EncoderBlock, self).__init__()
        n_hid = n_out // 4
        self.post_gain = 1 / (n_layers ** 2)

        self.id_path = nn.Conv2D(
            n_in, n_out, 1) if n_in != n_out else Identity()
        self.res_path = nn.Sequential(
            ('relu_1', nn.ReLU()),
            ('conv_1', nn.Conv2D(n_in,  n_hid, 3, padding=1)),
            ('relu_2', nn.ReLU()),
            ('conv_2', nn.Conv2D(n_hid, n_hid, 3, padding=1)),
            ('relu_3', nn.ReLU()),
            ('conv_3', nn.Conv2D(n_hid, n_hid, 3, padding=1)),
            ('relu_4', nn.ReLU()),
            ('conv_4', nn.Conv2D(n_hid, n_out, 1)))

    def forward(self, x):
        return self.id_path(x) + self.post_gain * self.res_path(x)


class Encoder(nn.Layer):
    def __init__(self, group_count=4, n_hid=256, n_blk_per_group=2, input_channels=3, vocab_size=8192):
        super(Encoder, self).__init__()
        blk_range = range(n_blk_per_group)
        n_layers = group_count * n_blk_per_group

        self.blocks = nn.Sequential(
            ('input', nn.Conv2D(input_channels, 1 * n_hid, 7, padding=3)),
            ('group_1', nn.Sequential(
                *[(f'block_{i + 1}', EncoderBlock(1 * n_hid, 1 *
                                                  n_hid, n_layers=n_layers)) for i in blk_range],
                ('pool', nn.MaxPool2D(kernel_size=2)),
            )),
            ('group_2', nn.Sequential(
                *[(f'block_{i + 1}', EncoderBlock(1 * n_hid if i == 0 else 2 *
                                                  n_hid, 2 * n_hid, n_layers=n_layers)) for i in blk_range],
                ('pool', nn.MaxPool2D(kernel_size=2)),
            )),
            ('group_3', nn.Sequential(
                *[(f'block_{i + 1}', EncoderBlock(2 * n_hid if i == 0 else 4 *
                                                  n_hid, 4 * n_hid, n_layers=n_layers)) for i in blk_range],
                ('pool', nn.MaxPool2D(kernel_size=2)),
            )),
            ('group_4', nn.Sequential(
                *[(f'block_{i + 1}', EncoderBlock(4 * n_hid if i == 0 else 8 *
                                                  n_hid, 8 * n_hid, n_layers=n_layers)) for i in blk_range],
            )),
            ('output', nn.Sequential(
                ('relu', nn.ReLU()),
                ('conv', nn.Conv2D(8 * n_hid, vocab_size, 1)),
            )),
        )

    def forward(self, x):
        return self.blocks(x)


class DecoderBlock(nn.Layer):
    def __init__(self, n_in, n_out, n_layers):
        super(DecoderBlock, self).__init__()
        n_hid = n_out // 4
        self.post_gain = 1 / (n_layers ** 2)

        self.id_path = nn.Conv2D(
            n_in, n_out, 1) if n_in != n_out else Identity()
        self.res_path = nn.Sequential(
            ('relu_1', nn.ReLU()),
            ('conv_1', nn.Conv2D(n_in,  n_hid, 1)),
            ('relu_2', nn.ReLU()),
            ('conv_2', nn.Conv2D(n_hid, n_hid, 3, padding=1)),
            ('relu_3', nn.ReLU()),
            ('conv_3', nn.Conv2D(n_hid, n_hid, 3, padding=1)),
            ('relu_4', nn.ReLU()),
            ('conv_4', nn.Conv2D(n_hid, n_out, 3, padding=1)))

    def forward(self, x):
        return self.id_path(x) + self.post_gain * self.res_path(x)


class Decoder(nn.Layer):
    def __init__(self, group_count=4, n_init=128, n_hid=256, n_blk_per_group=2, output_channels=3, vocab_size=8192):
        super(Decoder, self).__init__()

        blk_range = range(n_blk_per_group)
        n_layers = group_count * n_blk_per_group

        self.blocks = nn.Sequential(
            ('input', nn.Conv2D(vocab_size, n_init, 1)),
            ('group_1', nn.Sequential(
                *[(f'block_{i + 1}', DecoderBlock(n_init if i == 0 else 8 *
                                                  n_hid, 8 * n_hid, n_layers=n_layers)) for i in blk_range],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            )),
            ('group_2', nn.Sequential(
                *[(f'block_{i + 1}', DecoderBlock(8 * n_hid if i == 0 else 4 *
                                                  n_hid, 4 * n_hid, n_layers=n_layers)) for i in blk_range],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            )),
            ('group_3', nn.Sequential(
                *[(f'block_{i + 1}', DecoderBlock(4 * n_hid if i == 0 else 2 *
                                                  n_hid, 2 * n_hid, n_layers=n_layers)) for i in blk_range],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            )),
            ('group_4', nn.Sequential(
                *[(f'block_{i + 1}', DecoderBlock(2 * n_hid if i == 0 else 1 *
                                                  n_hid, 1 * n_hid, n_layers=n_layers)) for i in blk_range],
            )),
            ('output', nn.Sequential(
                ('relu', nn.ReLU()),
                ('conv', nn.Conv2D(1 * n_hid,
                                   2 * output_channels, 1)),
            )),
        )

    def forward(self, x):
        return self.blocks(x)
