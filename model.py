import torch
import torch.nn as nn

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = oup // ratio
        new_channels = oup - init_channels
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        self.use_shortcut = stride == 1 and inp == oup
        self.ghost1 = GhostModule(inp, hidden_dim, kernel_size=1, relu=True)
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size-1)//2, groups=hidden_dim, bias=False)
            self.bn_dw = nn.BatchNorm2d(hidden_dim)
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, hidden_dim // 4, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 4, hidden_dim, 1, bias=False),
                nn.Sigmoid()
            )
        else:
            self.se = None
        self.ghost2 = GhostModule(hidden_dim, oup, kernel_size=1, relu=False)
        if not self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size, stride, (kernel_size-1)//2, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.bn_dw(self.conv_dw(x))
        if self.se is not None:
            x = x * self.se(x)
        x = self.ghost2(x)
        return x + residual if self.use_shortcut else x + self.shortcut(residual)

class GhostFaceNet(nn.Module):
    def __init__(self, num_classes, width_mult=1.0):
        super(GhostFaceNet, self).__init__()
        self.cfgs = [
            [3, 16, 16, 0, 1], [3, 48, 24, 0, 2], [3, 72, 24, 0, 1],
            [5, 72, 40, 1, 2], [5, 120, 40, 1, 1], [3, 240, 80, 0, 2],
            [3, 200, 80, 0, 1], [3, 184, 80, 0, 1], [3, 184, 80, 0, 1],
            [3, 480, 112, 1, 1], [3, 672, 112, 1, 1], [5, 672, 160, 1, 2],
            [5, 960, 160, 0, 1], [5, 960, 160, 1, 1], [5, 960, 160, 0, 1], [5, 960, 160, 1, 1]
        ]
        output_channel = int(16 * width_mult)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel
        stages = []
        for k, exp_size, c, use_se, s in self.cfgs:
            out_c = int(c * width_mult)
            hid_c = int(exp_size * width_mult)
            stages.append(GhostBottleneck(input_channel, hid_c, out_c, k, s, use_se))
            input_channel = out_c
        self.blocks = nn.Sequential(*stages)
        output_channel = int(960 * width_mult)
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.act2 = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_final = nn.Conv2d(output_channel, 512, 1, 1, 0, bias=True)
        self.embedding = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x = self.blocks(x)
        x = self.act2(self.bn2(self.conv_head(x)))
        x = self.global_pool(x)
        x = self.conv_final(x).view(x.size(0), -1)
        embeddings = self.bn3(self.embedding(x))
        logits = self.classifier(embeddings)
        return embeddings, logits