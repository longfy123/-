# -*- coding: utf-8 -*-
import os, sys
import torch
import argparse
import numpy as np
import torch.utils.data
from easydict import EasyDict as edict
from timeit import default_timer as timer

from utils.eval import Metric
from utils.gpu_dispatch import GPU
from utils.common_utils import dir_check, to_device, ws, unfold_dict, dict_merge, GpuId2CudaId, Logger

from algorithm.dataset import CleanDataset, TrafficDataset
from algorithm.diffstg.model import DiffSTG, save2file

def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


# for tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard'))
except:
    pass

def get_params():
    parser = argparse.ArgumentParser(description='Entry point of the code')

    # model
    parser.add_argument("--epsilon_theta", type=str, default='UGnet')
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--N", type=int, default=50)  
    parser.add_argument("--beta_schedule", type=str, default='quad')  # uniform, quad
    parser.add_argument("--beta_end", type=float, default=0.1)
    parser.add_argument("--sample_steps", type=int, default=20)  
    parser.add_argument("--ss", type=str, default='ddpm') #help='sample strategy', ddpm, multi_diffusion, one_diffusion
    parser.add_argument("--T_h", type=int, default=12)  
    parser.add_argument("--T_p", type=int, default=1)  

    # eval
    parser.add_argument('--n_samples', type=int, default=1)

    # train
    parser.add_argument("--is_train", type=bool, default=True) # train or evaluate
    parser.add_argument("--data", type=str, default='shanghai')
    parser.add_argument("--data_type", type=str, default='residual', choices=['original', 'trend', 'seasonal', 'residual'])  
    parser.add_argument("--mask_ratio", type=float, default=0.0) # mask of history data
    parser.add_argument("--is_test", type=bool, default=False)  
    parser.add_argument("--nni", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=0.0005)  
    parser.add_argument("--batch_size", type=int, default=1)  

    args, _ = parser.parse_known_args()
    return args

def default_config(data='AIR_BJ'):
    config = edict()
    config.PATH_MOD = ws + '/output/model/'
    config.PATH_LOG = ws + '/output/log/'
    config.PATH_FORECAST = ws + '/output/forecast/'

    # Data Config
    config.data = edict()
    config.data.name = data
    config.data.path = ws + '/data/dataset/'
    config.data.data_type = 'original'  

    config.data.feature_file = config.data.path + config.data.name + '/flow.npy'
    config.data.spatial = config.data.path + config.data.name + '/adj.npy'
    config.data.num_recent = 1

    if config.data.name == 'shanghai':
        config.data.num_features = 1
        config.data.num_vertices = 1042
        config.data.points_per_hour = 2
        config.data.total_timesteps = 1488

        
        window_size = 12
        n_his = 12
        len_test = 8 * window_size + n_his
        len_val = 4 * window_size + n_his
        len_reserved = 16 * window_size + n_his
        len_train = config.data.total_timesteps - len_reserved - len_val - len_test
        len_train_full = len_train + len_reserved

        config.data.val_start_idx = len_train_full
        config.data.test_start_idx = len_train_full + len_val

    if config.data.name == 'nanjing':
        config.data.num_features = 1
        config.data.num_vertices = 1000
        config.data.points_per_hour = 2
        config.data.total_timesteps = 2784

        window_size = 12
        n_his = 12
        len_test = 8 * window_size + n_his
        len_val = 4 * window_size + n_his
        len_reserved = 16 * window_size + n_his
        len_train = config.data.total_timesteps - len_reserved - len_val - len_test
        len_train_full = len_train + len_reserved

        config.data.val_start_idx = len_train_full
        config.data.test_start_idx = len_train_full + len_val

    gpu_id = GPU().get_usefuel_gpu(max_memory=6000, condidate_gpu_id=[0,1,2,3,4,6,7,8])
    config.gpu_id = gpu_id
    if gpu_id != None:
        cuda_id = GpuId2CudaId(gpu_id)
        torch.cuda.set_device(f"cuda:{cuda_id}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model config
    config.model = edict()

    config.model.T_p = 1
    config.model.T_h = 12
    config.model.V = config.data.num_vertices
    config.model.F = config.data.num_features
    config.model.week_len = 7
    config.model.day_len = config.data.points_per_hour * 24
    config.model.device = device
    config.model.d_h = 32

    # config for diffusion model
    config.model.N = 50
    config.model.sample_steps = 20
    config.model.epsilon_theta = 'UGnet'
    config.model.is_label_condition = True
    config.model.beta_end = 0.02
    config.model.beta_schedule = 'quad'
    config.model.sample_strategy = 'ddpm'

    config.n_samples = 2

    # config for UGnet
    config.model.channel_multipliers = [1, 2]  # The list of channel numbers at each resolution.
    config.model.supports_len = 2

    # training config
    config.model_name = 'DiffSTG'
    config.is_test = False  # Whether run the code in the test mode
    config.epoch = 100  # Number of max training epoch
    config.optimizer = "adam"
    config.lr = 0.0005
    config.batch_size = 16  
    config.wd = 1e-5
    config.early_stop = 10
    config.start_epoch = 0
    config.device = device
    config.logger = Logger()

    if not os.path.exists(config.PATH_MOD):
        os.makedirs(config.PATH_MOD)
    config.PATH_LOG = ws + '/output/log/'
    if not os.path.exists(config.PATH_LOG):
        os.makedirs(config.PATH_LOG)
    if not os.path.exists(config.PATH_FORECAST):
        os.makedirs(config.PATH_FORECAST)
    return config

def evals(model, data_loader, epoch, metric, config, clean_data, mode='Test'):
    setup_seed(2022)

    y_pred, y_true, time_lst = [], [], []
    metrics_future = Metric(T_p=config.model.T_p)
    metrics_history = Metric(T_p=config.model.T_h)
    model.eval()

    samples, targets = [], []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i > 0 and config.is_test: break
            time_start = timer()

            future, history, pos_w, pos_d = to_device(batch, config.device)

            x = torch.cat((history, future), dim=1).to(config.device)
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device)
            targets.append(x.cpu())
            x = x.transpose(1, 3)
            x_masked = x_masked.transpose(1, 3)

            n_samples = 1 if mode == 'Val' else config.n_samples
            x_hat = model((x_masked, pos_w, pos_d), n_samples)
            samples.append(x_hat.transpose(2,4).cpu())

            if x_hat.shape[-1] != (config.model.T_h + config.model.T_p): x_hat = x_hat.transpose(2,4)

            time_lst.append((timer() - time_start))
            x, x_hat = clean_data.reverse_normalization(x), clean_data.reverse_normalization(x_hat)
            x_hat = x_hat.detach()
            f_x, f_x_hat = x[:,:,:,-config.model.T_p:], x_hat[:,:,:,:,-config.model.T_p:]

            _y_true_ = f_x.transpose(1, 3).cpu().numpy()
            _y_pred_ = f_x_hat.transpose(2, 4).cpu().numpy()
            _y_pred_ = np.clip(_y_pred_, 0, np.inf)
            metrics_future.update_metrics(_y_true_, _y_pred_)

            y_pred.append(_y_pred_)
            y_true.append(_y_true_)

            h_x, h_x_hat = x[:, :, :, :config.model.T_h], x_hat[:, :, :, :, :config.model.T_h]
            _y_true_ = h_x.transpose(1, 3).cpu().numpy()
            _y_pred_ = h_x_hat.transpose(2, 4).cpu().numpy()
            _y_pred_ = np.clip(_y_pred_, 0, np.inf)
            metrics_history.update_metrics(_y_true_, _y_pred_)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    time_cost = np.sum(time_lst)
    metric.update_metrics(y_true, y_pred)
    metric.update_best_metrics(epoch=epoch)
    metric.metrics['time'] = time_cost

    if mode == 'test': # save the prediction result to file
        samples = torch.cat(samples, dim=0)[:50]
        targets = torch.cat(targets, dim=0)[:50]
        observed_flag = torch.ones_like(targets) #(B, T, V, F)
        evaluate_flag = observed_flag
        evaluate_flag[:, -config.model.T_p:, :, :] = 1
        import pickle
        with open (config.forecast_path, 'wb') as f:
            pickle.dump([samples, targets, observed_flag, evaluate_flag], f)

        message = f"predict_path = '{config.forecast_path}'"
        config.logger.message_buffer += f"{message}\n"
        config.logger.write_message_buffer()


    if config.nni: nni.report_intermediate_result(metric.metrics['mae'])

    # log of performance in future prediction
    if metric.best_metrics['epoch'] == epoch:
        message = f" |[{metric.metrics['mae']:<7.2f}{metric.metrics['rmse']:<7.2f}]"
    else:
        message = f" | {metric.metrics['mae']:<7.2f}{metric.metrics['rmse']:<7.2f}"
    print(message, end='', flush=False)
    config.logger.message_buffer += message

    # log of performance in historical prediction
    message = f" | {metrics_history.metrics['mae']:<7.2f}{metrics_history.metrics['rmse']:<7.2f}{time_cost:<5.2f}s"
    print(message, end='\n', flush=False)
    config.logger.message_buffer += f"{message}\n"

    # write log message buffer
    config.logger.write_message_buffer()

    torch.cuda.empty_cache()
    return metric


from pprint import  pprint
def main(params: dict):
    # torch.manual_seed(2022)
    setup_seed(2022)
    torch.set_num_threads(2)
    config = default_config(params['data'])

    config.is_test = params['is_test']
    config.nni = params['nni']
    config.lr = params['lr']
    config.batch_size = params['batch_size']
    config.mask_ratio = params['mask_ratio']
    config.data.data_type = params.get('data_type', 'original')  

    # model
    config.model.N = params['N']
    config.T_h = config.model.T_h = params['T_h']
    config.T_p = config.model.T_p = params.get('T_p', params['T_h'])  
    config.model.epsilon_theta =  params['epsilon_theta']
    config.model.sample_steps = params['sample_steps']
    config.model.d_h = params['hidden_size']
    config.model.C = params['hidden_size']
    config.model.n_channels = params['hidden_size']
    config.model.beta_end = params['beta_end']
    config.model.beta_schedule = params["beta_schedule"]
    config.model.sample_strategy = params["ss"]
    config.n_samples = params['n_samples']

    if config.model.sample_steps > config.model.N:
        print('sample steps large than N, exit')
        # nni.report_intermediate_result(50)
        nni.report_final_result(50)
        return 0


    config.trial_name = '+'.join([f"{v}" for k, v in params.items()])
    config.log_path = '/dev/null'

    pprint(config)
    dir_check(config.log_path)
    config.logger.open(config.log_path, mode="w")
    #log parameters
    config.logger.write(config.__str__()+'\n', is_terminal=False)

    #  data pre-processing
    # print('\n1. data pre-processing ...')
    clean_data = CleanDataset(config)
    config.model.A = clean_data.adj

    model = DiffSTG(config.model)
    model = model.to(config.device)

    # Load training dataset
    train_dataset = TrafficDataset(clean_data, (0 + config.model.T_p, config.data.val_start_idx - config.model.T_p + 1), config)
    train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size, shuffle=True, pin_memory=True)

    _step = config.model.T_h + config.model.T_p
    val_start = config.data.val_start_idx + config.model.T_p
    val_dataset = TrafficDataset(clean_data, (val_start, val_start + 4 * _step), config)
    val_loader = torch.utils.data.DataLoader(val_dataset, 16, shuffle=False)

    test_start = config.data.test_start_idx + config.model.T_p
    test_dataset = TrafficDataset(clean_data, (test_start, test_start + 8 * _step), config)
    test_loader = torch.utils.data.DataLoader(test_dataset, 16, shuffle=False)


    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # metrics in val, and test dataset, note that we cannot evaluate the performance in the train dataset
    metrics_val = Metric(T_p=config.model.T_h + config.model.T_p)

    model_path = config.PATH_MOD + config.trial_name + model.model_file_name()
    config.model_path = model_path
    config.logger.write(f"model path:{model_path}\n", is_terminal=False)
    print('model_path:', model_path)
    dir_check(model_path)

    config.forecast_path = forecast_path = config.PATH_FORECAST + config.trial_name + '.pkl'
    config.logger.write(f"forecast_path:{model_path}\n", is_terminal=False)
    print('forecast_path:', forecast_path)
    dir_check(forecast_path)


    # log model architecture
    print(model)
    config.logger.write(model.__str__())

    # log training process
    config.logger.write(f'Num_of_parameters:{sum([p.numel() for p in model.parameters()])}\n', is_terminal=True)
    message = "      |---Train--- |---Val Future-- -|-----Val History----|\n"
    config.logger.write(message, is_terminal=True)

    message = "Epoch | Loss  Time | MAE     RMSE    |  MAE    RMSE   Time|\n" #f"{'Type':^5}{'Epoch':^5} | {'MAE':^7}{'RMSE':^7}{'MAPE':^7}
    config.logger.write(message, is_terminal=True)


    train_start_t = timer()
    # Train and sample the data
    for epoch in range(config.epoch):
        if not params['is_train']: break
        if epoch > 1 and config.is_test: break

        n, avg_loss, time_lst = 0, 0, []
        # train diffusion model
        for i, batch in enumerate(train_loader):
            if i > 3 and config.is_test:break
            time_start =  timer()
            future, history, pos_w, pos_d = batch # future:(B, T_p, V, F), history: (B, T_h, V, F)

            # get x0
            x = torch.cat((history, future), dim=1).to(config.device) #  (B, T, V, F)

            # get x0_masked
            mask =  torch.randint_like(history, low=0, high=100) < int(config.mask_ratio * 100)# mask the history in a ratio with mask_ratio
            history[mask] = 0
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device) # (B, T, V, F)

            # reshape
            x = x.transpose(1,3) # (B, F, V, T)
            x_masked = x_masked.transpose(1,3) # (B, F, V, T)

            # loss calculate
            loss = 10 * model.loss(x, (x_masked, pos_w, pos_d))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate the moving average training loss
            n += 1
            avg_loss = avg_loss * (n - 1) / n + loss.item() / n

            time_lst.append((timer() - time_start))
            message = f"{i / len(train_loader) + epoch:6.1f}| {avg_loss:0.3f} {np.sum(time_lst):.1f}s"
            print('\r' + message, end='', flush=True)

        config.logger.message_buffer += message

        try:
            writer.add_scalar('train/loss', avg_loss, epoch)
        except:
            pass

        if epoch >= config.start_epoch:
            evals(model, val_loader, epoch, metrics_val, config, clean_data, mode='Val')
            scheduler.step(metrics_val.metrics['mae'])

        if metrics_val.best_metrics['epoch'] == epoch:
            torch.save(model, model_path)

        if epoch - metrics_val.best_metrics['epoch'] > config.early_stop: break  # Early_stop


    try:
        model = torch.load(model_path, map_location=config.device)
    except Exception as err:
        pass

    # conduct multiple-samples, then report the best
    metric_lst = []
    for sample_strategy, sample_steps in [('ddim_multi', 40)]:
        if sample_steps > config.model.N: break

        config.model.sample_strategy = sample_strategy
        config.model.sample_steps = sample_steps

        model.set_ddim_sample_steps(sample_steps)
        model.set_sample_strategy(sample_strategy)

        metrics_test = Metric(T_p=config.model.T_h + config.model.T_p)
        evals(model, test_loader, epoch, metrics_test, config, clean_data, mode='test')
        m = metrics_test.metrics

        params = unfold_dict(config)
        params = dict_merge([params, metrics_test.to_dict()])
        params['best_epoch'] = metrics_val.best_metrics['epoch']
        params['model'] = config.model.epsilon_theta
        save2file(params)
        metric_lst.append(metrics_test.metrics['mae'])

    # rename log file
    log_file, log_name = os.path.split(config.log_path)
    new_log_path = os.path.join(log_file, f"[{config.data.name}]mae{min(metric_lst):7.2f}+{log_name}")
    import shutil
    # os.rename(config.log_path, new_log_path)
    shutil.copy(config.log_path, new_log_path)
    config.log_path = new_log_path

    try:
        writer.close()
    except:
        pass

    # nni.report_final_result(min(metric_lst))


# data.name	model	model.N	model.epsilon_theta	model.d_h	model.T_h	model.T_p	model.sample_strategy
# PEMS08	UGnet	200	    UGnet	            32	        12	        12	        ddpm

if __name__ == '__main__':

    import nni
    import logging

    logger = logging.getLogger('training')

    print('GPU:', torch.cuda.current_device())
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
