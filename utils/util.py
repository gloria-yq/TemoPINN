"""通用工具函数."""

import logging

import numpy as np
from sklearn import metrics


def get_logger(log_name='log.txt'):
    logger = logging.getLogger('weather_pinn')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_name is not None:
        file_handler = logging.FileHandler(log_name, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def eval_metrix(pred_label, true_label):
    mae = metrics.mean_absolute_error(true_label, pred_label)
    mape = metrics.mean_absolute_percentage_error(true_label, pred_label)
    mse = metrics.mean_squared_error(true_label, pred_label)
    rmse = np.sqrt(mse)
    return [mae, mape, mse, rmse]


def write_to_txt(txt_name, txt):
    with open(txt_name, 'a', encoding='utf-8') as f:
        f.write(txt)
        f.write('\n')
