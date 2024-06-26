import torch
from torchviz import make_dot


class GenericLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        
    def forward(self, X):
        return self.model(X)

class Layer1(GenericLayer):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.Conv2d(in_channels=out_ch, out_channels=out_ch, padding='same', kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.Dropout(0.5)
        )

class Layer2(GenericLayer):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.Dropout(0.5)
        )

class Layer3(GenericLayer):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_ch, out_features=out_ch),
            torch.nn.BatchNorm1d(out_ch)
        )

class Architecture(GenericLayer):
    def __init__(self, args):
        super().__init__()

        x = torch.randn(args['image_shape'])
        self.model =  torch.nn.Sequential()
        in_ch1 = args['in_ch1']
        out_ch1 = args['out_ch1']
        in_ch2 = args['in_ch2']
        out_ch2 = args['out_ch2']

        def addLayers(layername, layerclass, in_ch, out_ch, x):
            for i in range(args[layername]):
                layer = layerclass(in_ch,out_ch)
                self.model.add_module(f'{layername}_{i}', layer)
                x = layer(x)
                in_ch = x.shape[1]
                if layername == 'layer3': out_ch = out_ch//2
                else: out_ch = 2*out_ch
            return (in_ch, out_ch, x)
        
        in_ch1, out_ch1, x = addLayers('layer1', Layer1, in_ch1, out_ch1, x)
        in_ch1, out_ch1, x = addLayers('layer2', Layer2, in_ch1, out_ch1, x)
        layer = torch.nn.Flatten()
        self.model.add_module('flatten', layer)
        x = layer(x)
        in_ch1 = x.shape[1]
        out_ch1 = in_ch2
        in_ch1, out_ch1, x = addLayers('layer3', Layer3, in_ch1, out_ch1, x)
        layer = torch.nn.Linear(in_features=in_ch1, out_features=out_ch2)
        self.model.add_module('Linear', torch.nn.Linear(in_features=in_ch1, out_features=out_ch2))
        x = layer(x)
        layer = torch.nn.Softmax(dim=1)
        self.model.add_module('Softmax', layer)
        x = layer(x)

        make_dot(x, params=dict(list(self.model.named_parameters()))).render(args['arct_name']+'/graph', format="png")
