import os
import random
import numpy as np
import torch
import argparse

from load_data import load_dataset, split_dataset
from attack_model import URMA

# Locking random seed
def seed_setting(seed=41):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# seed_setting()

parser = argparse.ArgumentParser()
# dataset
parser.add_argument('--dataset', dest='dataset', default='derment', choices=['derment', 'isic'])
parser.add_argument('--dataset_path', dest='dataset_path', default='train')
# attacked model
parser.add_argument('--method', dest='method', default='CSQ', choices=['DPSH', 'CSQ'])
parser.add_argument('--attacked_models_path', dest='attacked_models_path', default='attacked_models/')
# output
parser.add_argument('--output_dir', dest='output_dir', default='output')
parser.add_argument('--output_path', dest='output_path', default='outputs/')
# detail setting
parser.add_argument('--bit', dest='bit', type=int, default=64)
parser.add_argument('--gpu', dest='gpu', type=str, default='0', choices=['0', '1', '2', '3'])
# knockoff model
parser.add_argument('--knockoff_bit', dest='kb', type=int, default=64)
parser.add_argument('--knockoff_epochs', dest='ke', type=int, default=20)
parser.add_argument('--knockoff_batch_size', dest='kbz', type=int, default=48)
parser.add_argument('--knockoff_text_learning_rate', dest='ktlr', type=float, default=1e-3)
parser.add_argument('--knockoff_image_learning_rate', dest='kilr', type=float, default=1e-4)
# perturbation model
parser.add_argument('--perturbation_epochs', dest='pe', type=int, default=20)
parser.add_argument('--perturbation_batch_size', dest='pbz', type=int, default=24)
parser.add_argument('--perturbation_learning_rate', dest='plr', type=float, default=1e-4)
# attack model
parser.add_argument('--attack_epochs', dest='ae', type=int, default=100)
parser.add_argument('--attack_batch_size', dest='abz', type=int, default=24)
parser.add_argument('--attack_learning_rate', dest='alr', type=float, default=1e-4)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# data processing
seed_setting()
dataset = load_dataset(args.dataset,args.dataset_path,num_label=23)
train_set, val_set, database_set = split_dataset(dataset)

print('Train length:', len(train_set), 'Test length:', len(val_set), 'Database length:', len(database_set))


# black-box attack
URMA = URMA(args)
URMA.test_attacked_model( val_set, database_set)
URMA.train_knockoff(train_set)
URMA.test_knockoff(val_set,database_set)
URMA.train_attack_model(train_set, database_set)


