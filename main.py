import copy
import json
import os
import random
import time
import warnings
from absl import app, flags
from tqdm import trange
import torch
import numpy as np

from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.utils import make_grid, save_image
from torchvision import transforms

from diffusion import GaussianDiffusionSampler
from model.model import UNet
from utils.augmentation import *
from dataset import ImbalanceCIFAR100, ImbalanceCIFAR10
from utils.augmentation import KarrasAugmentationPipeline


FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load model.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
flags.DEFINE_bool('improve', False, help='use improved diffusion network implemented by OpenAI')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help='gradient norm clipping')
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help='ema decay rate')
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_bool('conditional', False, help='conditional generation')
flags.DEFINE_bool('weight', False, help='reweight')
flags.DEFINE_bool('cotrain', False, help='cotrain with an adjusted classifier or not')
flags.DEFINE_bool('logit', False, help='use logit adjustment or not')
flags.DEFINE_bool('augm', False, help='whether to use ADA augmentation')
flags.DEFINE_bool('cfg', False, help='whether to train unconditional generation with with 10%  probability')
# Dataset
flags.DEFINE_string('root', './', help='path of dataset')
flags.DEFINE_string('data_type', 'cifar100', help='data type, must be in [cifar10, cifar100, cifar10lt, cifar100lt]')
flags.DEFINE_float('imb_factor', 0.01, help='imb_factor for long tail dataset')
flags.DEFINE_float('num_class', 0, help='number of class of the pretrained model')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/', help='log directory')
flags.DEFINE_integer('sample_size', 64, 'sampling size of images')
flags.DEFINE_integer('sample_step', 10000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 100000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_integer('private_num_images', 0, help='the number of private images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_string('sample_name', 'saved', help='name for a set of samples to be saved or to be evaluated')
flags.DEFINE_bool('sampled', False, help='evaluate sampled images')
flags.DEFINE_string('sample_method', 'cfg', help='sampling method, must be in [cfg, cond, uncond]')
flags.DEFINE_float('omega', 0.0, help='guidance strength for cfg sampling method')
flags.DEFINE_bool('prd', True, help='evaluate precision and recall (F_beta), only evaluated with 50k samples')
flags.DEFINE_bool('improved_prd', True, help='evaluate improved precision and recall, only evaluated with 50k samples')
# CBDM hyperparameters
flags.DEFINE_bool('cb', False, help='train with class-balancing(adjustment) loss')
flags.DEFINE_float('tau', 1.0, help='weight for the class-balancing(adjustment) loss')
# CBDM finetuning mechanism
flags.DEFINE_bool('finetune', False, help='finetuned based on a pretrained model')
flags.DEFINE_string('finetuned_logdir', '', help='logdir for the new model, where FLAGS.logdir will be the folder for \
                     the pretrained model')
flags.DEFINE_integer('ckpt_step', 0, help='step to reload the pretained checkpoint')
# CBDM-ET hyperparameters
flags.DEFINE_integer('S', 5, help='step for CBDM')
flags.DEFINE_integer('cut_off_value', 0, help='')
flags.DEFINE_float('fix_scale', 1.0, help='')
flags.DEFINE_float('r', 0.0, help='')
flags.DEFINE_integer('seed', 0, help='')
flags.DEFINE_integer('window_size', 0, help='')
flags.DEFINE_bool('shift_time_step', False, help='')
flags.DEFINE_string('sample_type', 'ddim', help='')
flags.DEFINE_string('org_dir', '', help='')
flags.DEFINE_float('FID', 10000, help='')
flags.DEFINE_float('p', 0.9, help='')
flags.DEFINE_integer('n_intervals', 5, help='')
flags.DEFINE_integer('t_cutoff', 100, help='')
flags.DEFINE_string('output_logdir', 'output.npz', help='')
device = torch.device('cuda:0')


def uniform_sampling(n, N, k):
    return np.stack([np.random.randint(int(N/n)*i, int(N/n)*(i+1), k) for i in range(n)])


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def evaluate(sampler, model, sampled):
    if not sampled:
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            images = []
            labels = []
            desc = 'generating images'
            image_count = 0
            for i in trange(0, FLAGS.num_images, FLAGS.batch_size,desc=desc):
                batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
                x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
                if FLAGS.sample_type == 'DDPM':
                    batch_images, batch_labels = sampler(x_T.to(device),
                                                         omega=FLAGS.omega,
                                                         method=FLAGS.sample_method)
                elif FLAGS.sample_type in ['ddim','s_pndm','f_pndm','euler','ddpm','deis','dpm-solver']:
                    batch_images, batch_labels = sampler.sample(S=FLAGS.S,
                                         batch_size=batch_size,
                                         method=FLAGS.sample_type,
                                         # shape=(3,128,128),
                                         shape=(x_T.shape[1:]),
                                         )

                batch_images = ((batch_images.cpu() + 1) / 2) * 255
                batch_images = batch_images.clamp(0, 255)
                batch_images = batch_images.to(torch.long)
                batch_images = batch_images.to(torch.uint8)
                images.append(batch_images)

                if FLAGS.sample_method!='uncond' and batch_labels is not None:
                    labels.append(batch_labels.cpu())
        images = torch.cat(images, dim=0).numpy()
        end_time = time.time()
        elapsed_time = end_time - start_time

    images = np.transpose(images, (0, 2, 3, 1))
    output_npz = np.array(images)
    np.savez(FLAGS.output_logdir, output_npz)




def eval():
    if 'cifar100' in FLAGS.data_type:
        FLAGS.num_class = 100
    elif 'cifar10' in FLAGS.data_type:
        FLAGS.num_class = 10
    elif 'celeba' in FLAGS.data_type:
        FLAGS.num_class = 5

    print(FLAGS.num_class)
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,
        cond=FLAGS.conditional, augm=FLAGS.augm, num_class=FLAGS.num_class)

    if FLAGS.sample_type == 'DDPM':
        sampler = GaussianDiffusionSampler(
            model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.num_class, FLAGS.img_size, FLAGS.var_type, FLAGS.t_cutoff, FLAGS.n_intervals).to(device)

    if FLAGS.sample_type in ['ddim','s_pndm','f_pndm','euler','ddpm']:
        from sampler.pnm_solver1 import PNMSolver
        sampler = PNMSolver(
            model=model,
            device=device,
            diffusion_sampling_steps=FLAGS.T,
            beta_start=FLAGS.beta_1,
            beta_end=FLAGS.beta_T,
            schedule_type="linear",
            shift_time_step=FLAGS.shift_time_step,
            window_size=FLAGS.window_size,
            cut_off_value=FLAGS.cut_off_value,
            step_size=None,
            fix_scale=FLAGS.fix_scale,
            normalize_variance=False,
            eta=0,
            scale_method=True,
            num_class=FLAGS.num_class,
            batch_size=FLAGS.batch_size,
            omega=FLAGS.omega,
            r = FLAGS.r,
        )

    elif FLAGS.sample_type == "dpm-solver":
        from sampler.dpm_solver import DPMSolverSampler
        sampler = DPMSolverSampler(
            model=model,
            device=device,
            shift_time_step=FLAGS.shift_time_step,
            window_size=FLAGS.window_size,
            cut_off_value=FLAGS.cut_off_value,
            diffusion_sampling_steps=FLAGS.T,
            beta_start = FLAGS.beta_1,
            beta_end = FLAGS.beta_T,
            schedule_type="linear",
            num_class=FLAGS.num_class,
            batch_size=FLAGS.batch_size,
            omega=FLAGS.omega,
            r=FLAGS.r,
        )

    elif FLAGS.sample_type in ["deis","ipndm"]:
        if FLAGS.sample_type == "deis":
            method = 't_ab'
        else:
            method = 'ipndm'
        from sampler.deis_sampler import DEIS_Sampler
        sampler = DEIS_Sampler(
            model = model,
            device=device,
            diffusion_sampling_steps=FLAGS.T,
            beta_start = FLAGS.beta_1,
            beta_end=FLAGS.beta_T,
            schedule_type="linear",
            num_steps = FLAGS.S,
            shift_time_step=FLAGS.shift_time_step,
            window_size = FLAGS.window_size,
            cut_off_value = FLAGS.cut_off_value,
            num_class=FLAGS.num_class,
            omega=FLAGS.omega,
            r=FLAGS.r,
            method=method
        )
    else:
        print("please choose one")
    FLAGS.sample_name = FLAGS.org_dir
    FLAGS.sample_name = '{}_N{}'.format(FLAGS.org_dir, FLAGS.num_images)

    # load ema model (almost always better than the model) and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt_{}.pt'.format(FLAGS.ckpt_step)), map_location='cpu')

    # evaluate IS/FID
    if 'cifar100' in FLAGS.data_type:
        FLAGS.fid_cache = './stats/cifar100.train.npz'
    elif 'cifar10' in FLAGS.data_type:
        FLAGS.fid_cache = './stats/cifar10.train.npz'

    if not FLAGS.sampled:
        model.load_state_dict(ckpt['ema_model'])
    else:
        model = None

    evaluate(sampler, model, FLAGS.sampled)


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
       pass
    if FLAGS.eval:
        FLAGS.org_dir = FLAGS.sample_name
        eval()
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')



if __name__ == '__main__':
    app.run(main)




