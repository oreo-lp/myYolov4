import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class YOLOLayer(nn.Module):
    def __init__(self):
        super(YOLOLayer, self).__init__()

    def forward(self, x):
        return x


class Darknet(nn.Module):
    def __init__(self, cfg_path):
        super(Darknet, self).__init__()
        # parse cfg file
        self.cfg_path = cfg_path
        self.net_info, self.module_lst = self.parse_cfg()
        # build modules
        self.net_modules = self.build_modules()

    def forward(self, x):
        pass

    # parse cfg file
    def parse_cfg(self):
        # read cfg_path
        with open(self.cfg_path, 'r') as f:
            lines = f.read().splitlines()
            # remove null strings and annotation
            lines = list(filter(lambda x: len(x) > 0 and not x.startswith('#'), lines))

        module_lst = []

        for line in lines:
            # a new module
            if line.startswith('['):
                module_lst.append({})
                module_lst[-1]['type'] = line[1:-1].strip()
            # module info
            else:
                key, value = line.split('=')
                module_lst[-1][key.strip()] = value.strip()
        net_info = module_lst.pop(0)
        return net_info, module_lst

    # build modules
    def build_modules(self):
        inp_filters = 3  # the input channels of every layers
        oup_channels = []  # save every channels for shortcut and route layers
        net_modules = nn.ModuleList()  # modules container

        for index, module in enumerate(self.module_lst):
            layer = nn.Sequential()
            type_module = module['type']
            if 'convolutional' == type_module:
                # not every layers has bn
                try:
                    bn = int(module['batch_normalize'])
                except:
                    bn = 0
                filters, kernel_size, stride = int(module['filters']), int(module['size']), int(module['stride'])
                pad = (kernel_size - 1) // 2 if module['pad'] else 0
                layer.add_module('conv_%d' % index,
                                 nn.Conv2d(inp_filters, filters, kernel_size=kernel_size, stride=stride, padding=pad,
                                           bias=not bn))
                if bn:
                    layer.add_module('bn_%d' % index, nn.BatchNorm2d(num_features=filters))
                if 'mish' == module['activation']:
                    layer.add_module('mish_%d' % index, Mish())
                # update input channels
                inp_filters = filters

            elif 'upsample' == type_module:
                stride = int(module['stride'])
                layer.add_module('upsample_%d' % index, nn.Upsample(scale_factor=stride))

            elif 'maxpool' == type_module:
                kernel_size, stride = int(module['size']), int(module['stride'])
                layer.add_module('maxpool_%d' % index, nn.MaxPool2d(kernel_size=kernel_size, stride=stride))

            elif 'shortcut' == type_module:
                # add empty Layer
                layer.add_module('shortcut_%d' % index, EmptyLayer())
                # update input channels
                inp_filters = oup_channels[int(module['from'])]

            elif 'route' == type_module:
                layers_ = [int(x) for x in module['layers'].split(',')]
                # only one value
                if len(layers_) == 1:
                    inp_filters = oup_channels[layers_[0]]
                elif len(layers_) == 2:
                    inp_filters = oup_channels[layers_[0]] + oup_channels[layers_[1]]
                elif len(layers_) == 4:
                    inp_filters = oup_channels[layers_[0]] + oup_channels[layers_[1]] + oup_channels[layers_[2]] + \
                                  oup_channels[layers_[3]]
            elif 'yolo' == type_module:
                mask = [int(x) for x in module['mask'].split(',')]
                anchors = [int(x) for x in module['anchors'].split(',')]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]
                layer.add_module('yolo_%d' % index, YOLOLayer())

            # add inp_filter to oup_channels
            oup_channels.append(inp_filters)
            # add every layers to net modules
            net_modules.append(layer)

        return net_modules


if __name__ == '__main__':
    darknet = Darknet('./cfg/yolov4.cfg')
