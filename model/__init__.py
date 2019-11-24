
from .model import ft_net
from .model import Net
from .mobilenet import MobileNet

def build_model(args,num_calsses):
    print('model name is:',args.model_name)
    if args.model_name == 'resnet50':
        model = ft_net(num_calsses,1)
    elif args.model_name == 'mobile':
        model = MobileNet(num_calsses)
    elif args.model_name == 'tiny_resnet50':
        model = Net(num_calsses,reid=False)

    return model