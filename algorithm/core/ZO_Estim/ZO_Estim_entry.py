import torch

def split_model(model):
    modules = []
    for m in model.children():
        if isinstance(m, (torch.nn.Sequential,)):
            modules += split_model(m)
        elif hasattr(m, 'conv') and isinstance(m.conv, torch.nn.Sequential):
            modules += split_model(m.conv)
        else:
            modules.append(m)
    return modules

def split_named_model(model, parent_name=''):
    named_modules = {}
    for name, module in model.named_children():
    # for name, module in model.named_modules():    # Error: non-stop recursion
        if isinstance(module, torch.nn.Sequential):
            named_modules.update(split_named_model(module, parent_name + name + '.'))
        elif hasattr(module, 'conv') and isinstance(module.conv, torch.nn.Sequential):
            named_modules.update(split_named_model(module.conv, parent_name + name + '.conv.'))
        else:
            named_modules[parent_name + name] = module
    return named_modules

from .ZO_Estim_MC import ZO_Estim_MC

def build_ZO_Estim(config, model, obj_fn):
    if config.name == 'ZO_Estim_MC':
        ZO_Estim = ZO_Estim_MC(
            model = model, 
            obj_fn = obj_fn,

            sigma = config.sigma,
            n_sample  = config.n_sample,
            signSGD = config.signSGD,
            trainable_param_list = config.trainable_param_list,
            trainable_layer_list = config.trainable_layer_list,

            quantize_method = config.quantize_method,
            mask_method = config.mask_method,
            estimate_method = config.estimate_method,
            perturb_method = config.perturb_method,
            sample_method = config.sample_method,
            prior_method = config.prior_method
        )
        return ZO_Estim
    else:
        return NotImplementedError

def build_obj_fn(obj_fn_type, **kwargs):
    if obj_fn_type == 'classifier':
        obj_fn = build_obj_fn_classifier(**kwargs)
    elif obj_fn_type == 'classifier_layerwise':
        obj_fn = build_obj_fn_classifier_layerwise(**kwargs)
    elif obj_fn_type == 'classifier_acc':
        obj_fn = build_obj_fn_classifier_acc(**kwargs)
    else:
        return NotImplementedError
    return obj_fn

def build_obj_fn_classifier(data, target, model, criterion):
    def _obj_fn():
        y = model(data)
        return y, criterion(y, target)
    
    return _obj_fn

def build_obj_fn_classifier_acc(data, target, model, criterion):
    def _obj_fn():
        outputs = model(data)
        _, predicted = outputs.max(1)
        total = target.size(0)
        correct = predicted.eq(target).sum().item()
        err = 1 - correct / total

        return outputs, err
    
    return _obj_fn

def build_obj_fn_classifier_layerwise(data, target, model, criterion):
    split_modules_list = split_model(model)
    
    # if no attribute for _obj_fn: same as build_obj_fn_classifier
    def _obj_fn(starting_idx=0, ending_idx=None, input=None, return_loss_reduction='mean'):
        if ending_idx == None:
            ending_idx = len(split_modules_list)

        if starting_idx == 0:
            y = data
        else:
            assert input is not None
            y = input

        for i in range(starting_idx, ending_idx):
            y = split_modules_list[i](y)
        
        if return_loss_reduction == 'mean':
            criterion.reduction = 'mean'
            return y, criterion(y, target)
        elif return_loss_reduction == 'none':
            criterion.reduction = 'none'
            loss = criterion(y, target)
            criterion.reduction = 'mean'
            return y, loss
        elif return_loss_reduction == 'no_loss':
            return y
        else:
            raise NotImplementedError(f'Unknown {return_loss_reduction}')
    
    return _obj_fn