def build_stereo_depth(config):
    print('###################################')
    print(config.model_name)
    if config.model_name == 'psm_basic':
        from .psm_net import PSMNet
        from .losses import PSMBasicLoss
        model = PSMNet(config)
        criterion = PSMBasicLoss()
    elif config.model_name == 'psm_stackhourglass':
        from .psm_net import PSMNet
        from .losses import PSMDeepLoss
        model = PSMNet(config)
        if config.self_supervised:
            criterion = PSMDeepLossSelfSupervised(config)
        else:
            criterion = PSMDeepLoss(config)
    elif config.model_name == 'ga_net':
        from .ga_net import GANet
        from .losses import GALoss
        model = GANet(config)
        criterion = GALoss(config)
    elif config.model_name == 'aa_net':
        from .aa_net import AANet
        from .losses import AALoss
        model = AANet(config)
        if config.self_supervised:
            criterion = AALossSelfSupervised(config)
        else:
            criterion = AALoss(config)
    elif config.model_name == 'sttr':
        from .sttr_module.sttr import STTR
        from .sttr_module.loss import sttr_criterion
        from .losses import stereo_psmnet_loss
        model = STTR(config)
        criterion = []
        criterion.append(sttr_criterion(config, 4))
        criterion.append(stereo_psmnet_loss(config))
    elif config.model_name == 'pasm':
        from models.PASMnet import PASMnet
        model = PASMnet()
        criterion = None
    else:
        raise NotImplementedError("model name {} not supported".format(config.model_name))
    return model, criterion
