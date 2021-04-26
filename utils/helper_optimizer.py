import torch


def get_optimizer(model, config):
    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.learning_rate,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
    elif config.optimizer == 'adamW':
        # print(model)
        # cal_params = list(map(id, model.regression_head.cal.parameters()))
        # rest_params = filter(lambda x: id(x) not in cal_params, model.parameters())
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if
                        "backbone" not in n and "regression" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": 1e-4,
            },
            {
                "params": [p for n, p in model.named_parameters() if "regression" in n and p.requires_grad],
                "lr": 2e-4,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts,
                                      lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        # optimizer = torch.optim.AdamW(model.parameters(),
        #                              lr=config.learning_rate,
        #                              weight_decay=config.weight_decay)
    else:
        raise NotImplementedError("Optimizer {} not supported".format(config.optimizer))

    return optimizer

