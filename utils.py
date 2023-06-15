import logging
import os
import shutil
import datetime
import torch
import sys


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id, fs_prefix='./'):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(fs_prefix + 'output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def copy_all_files(file, output_dir):
    dir_src = os.path.dirname(file)
    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.py'):
            shutil.copy(os.path.join(dir_src, filename), output_dir)


def set_gpu(gpu):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    # exp_id = get_exp_id(__file__)
    exp_id = 'ebm_plot'
    output_dir = get_output_dir(exp_id, fs_prefix='../alienware_')
    print(exp_id)
    print(os.getcwd())
    print(__file__)
    print(os.path.basename(__file__))
    print(os.path.dirname(__file__))
    # copy_source(__file__, output_dir)
    copy_all_files(__file__, output_dir)
