# an image classification trainer
import os
import sys
import argparse
import time

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

from core.utils import dist
from core.model import build_mcu_model
from core.utils.config import configs, load_config_from_file, update_config_from_args, update_config_from_unknown_args
from core.utils.logging import logger
from core.dataset import build_dataset
from core.optimizer import build_optimizer
from core.ZO_Estim.ZO_Estim_entry import build_ZO_Estim
from core.trainer.cls_trainer import ClassificationTrainer
from core.builder.lr_scheduler import build_lr_scheduler

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('config', metavar='FILE', help='config file')
parser.add_argument('--run_dir', type=str, metavar='DIR', help='run directory')
parser.add_argument('--evaluate', action='store_true')


def build_config():  # separate this config requirement so that we can call main() in ray tune
    # support extra args here without setting in args
    args, unknown = parser.parse_known_args()

    load_config_from_file(args.config)
    update_config_from_args(args)
    update_config_from_unknown_args(unknown)


def main():
    dist.init()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    # assert configs.run_dir is not None
    if configs.run_dir is None:
        if configs.ZO_Estim.en is True:
            training_method = 'ZO'
        else:
            training_method = 'FO'
        configs.run_dir = os.path.join(
            "./runs",
            configs.data_provider.root.split('/')[-1],
            configs.net_config.net_name, 
            training_method,
            time.strftime("%Y%m%d-%H%M%S")+'-'+str(os.getpid())
        )
    
    os.makedirs(configs.run_dir, exist_ok=True)
    logger.init()  # dump exp config
    logger.info(str(os.getpid()))
    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{configs.run_dir}".')

    # set random seed
    torch.manual_seed(configs.manual_seed)
    torch.cuda.manual_seed_all(configs.manual_seed)

    # create dataset
    dataset = build_dataset()
    data_loader = dict()
    for split in dataset:
        sampler = torch.utils.data.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            seed=configs.manual_seed,
            shuffle=(split == 'train'))
        data_loader[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.data_provider.base_batch_size,
            sampler=sampler,
            num_workers=configs.data_provider.n_worker,
            pin_memory=True,
            drop_last=(split == 'train'),
        )

    # create model
   
    model = build_mcu_model()
    
    if configs.data_provider.load_model_path is not None:
        checkpoint = torch.load(configs.data_provider.load_model_path, map_location='cpu')
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        logger.info(f'Load model from {configs.data_provider.load_model_path}')

        if 'best_val' in checkpoint:
            logger.info('loaded best_val: %f' % checkpoint['best_val'])
    
    model = model.cuda()

    # for idx, m in enumerate(model.named_modules()):
    #     print(idx, '->', m)
    
    # for idx, (name, param) in enumerate(model.named_parameters()):
    #     print(idx, '->', name, param.shape)

    # for idx, (name, param) in enumerate(model.named_parameters()):
    #     print(idx, '->', name, param.numel())

    if dist.size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.local_rank()])  # , find_unused_parameters=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model)
    lr_scheduler = build_lr_scheduler(optimizer, len(data_loader['train']))

    if configs.ZO_Estim.en is True:
        obj_fn = None
        ZO_Estim = build_ZO_Estim(configs.ZO_Estim, model=model, obj_fn=obj_fn)
        # if configs.ZO_Estim.fc_bp == True:
        #     for name, param in model.named_parameters():
        #         if name in ('4.weight', '4.bias'):
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False
    else:
        ZO_Estim = None

    trainer = ClassificationTrainer(model, data_loader, criterion, optimizer, lr_scheduler, ZO_Estim)

    # kick start training
    if configs.resume:
        trainer.resume()  # trying to resume

    if configs.backward_config.enable_backward_config:
        from core.utils.partial_backward import parsed_backward_config, prepare_model_for_backward_config, \
            get_all_conv_ops
        configs.backward_config = parsed_backward_config(configs.backward_config, model)
        prepare_model_for_backward_config(model, configs.backward_config)
        logger.info(f'Getting backward config: {configs.backward_config} \n'
                    f'Total convs {len(get_all_conv_ops(model))}')

    if configs.evaluate:
        val_info_dict = trainer.validate()
        print(val_info_dict)
        return val_info_dict  # for ray tune
    else:
        val_info_dict = trainer.run_training()
        return val_info_dict  # for ray tune


if __name__ == '__main__':
    build_config()
    main()



# trainable_param_list = []

    # from core.utils.partial_backward import get_all_conv_ops_with_names
    # convs, names = get_all_conv_ops_with_names(model)
    # manual_weight_idx = [21,24,27,30,36,39]
    # for i_name, name in enumerate(names):  # from input to output
    #     if i_name in manual_weight_idx:  # the weight is updated for this layer
    #         trainable_param_list.append(name)
    
    # bias_name_list = []
    # n_bias_update = 12
    # for name, m in model.named_parameters():
    #     if 'bias' in name:
    #         bias_name_list.append(name)
    # bias_name_list = bias_name_list[-n_bias_update:]

    # n_weight_update = 12
    # weight_name_list = names[-12:]

    # print(trainable_param_list)
    # print(weight_name_list)
    # print(bias_name_list)