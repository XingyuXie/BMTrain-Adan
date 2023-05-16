import torch
from torch import Tensor
import math
from ..global_var import config
import torch.optim._functional as F
from . import _cuda as C
from .. import nccl
import inspect

from copy import deepcopy
from itertools import chain
from collections import defaultdict

def _single_tensor_adan(
    param: Tensor,
    grad: Tensor,
    neg_pre_grad: Tensor,
    exp_avg: Tensor,
    exp_avg_diff: Tensor,
    exp_avg_sq: Tensor,
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float
):

    # for memory saving, we use `neg_grad_or_diff`
    # to get some temp variable in a inplace way
    neg_grad_or_diff = neg_pre_grad
    neg_grad_or_diff.add_(grad)

    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
    exp_avg_diff.mul_(beta2).add_(neg_grad_or_diff,
                                    alpha=1 - beta2)  # diff_t

    neg_grad_or_diff.mul_(beta2).add_(grad)
    exp_avg_sq.mul_(beta3).addcmul_(neg_grad_or_diff,
                                    neg_grad_or_diff,
                                    value=1 - beta3)  # n_t

    denom = ((exp_avg_sq).sqrt() / bias_correction3_sqrt).add_(eps)
    step_size_diff = lr * beta2 / bias_correction2
    step_size = lr / bias_correction1
    param.addcdiv_(exp_avg, denom, value=-step_size)
    param.addcdiv_(exp_avg_diff, denom, value=-step_size_diff)
    param.div_(1 + lr * weight_decay)



class AdanOptimizer(torch.optim.Optimizer):
    """
        Implements a pytorch variant of Adan
        Adan was proposed in
        Adan: Adaptive Nesterov Momentum Algorithm for
            Faster Optimizing Deep Models[J].arXiv preprint arXiv:2208.06677, 2022.
        https://arxiv.org/abs/2208.06677
        Arguments:
            params (iterable): iterable of parameters to optimize or
                dicts defining parameter groups.
            lr (float, optional): learning rate. (default: 1e-3)
            betas (Tuple[float, float, flot], optional): coefficients used for
                first- and second-order moments. (default: (0.98, 0.92, 0.99))
            eps (float, optional): term added to the denominator to improve
                numerical stability. (default: 1e-8)
            weight_decay (float, optional): decoupled weight decay
                (L2 penalty) (default: 0)
    """
    _bmtrain_optimizer = True

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.98, 0.92, 0.99),
                 eps=1e-8,
                 weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(
                betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError('Invalid beta parameter at index 2: {}'.format(
                betas[2]))

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)


    def _on_justify_scale(self, old_scale, new_scale):
        delta = new_scale / old_scale
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state:
                    state = self.state[p]
                    if len(state) > 0:
                        state['exp_avg'] *= delta
                        state['exp_avg_diff'] *= delta
                        state['exp_avg_sq'] *= delta
                        state['neg_pre_grad'] *= delta

    @torch.no_grad()
    def step(self, closure=None, scale=1):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # update parameters
        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            for p in group['params']:
                if p.grad is not None and p.requires_grad:
                    if p.grad.is_sparse:
                        raise RuntimeError('Adan does not support sparse gradients, please consider SparseAdam instead')
                    if p.dtype not in [torch.float16, torch.float32]:
                        raise RuntimeError('Adan only supports fp32 or fp16 gradients')

                    state = self.state[p]
                
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros(p.size(), dtype=torch.float32, device=p.device) # on device
                        # Exponential moving average of gradient difference
                        state['exp_avg_diff'] = torch.zeros(p.size(), dtype=torch.float32, device=p.device) # on device
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros(p.size(), dtype=torch.float32, device=p.device)   # on device

                        if p.dtype == torch.half:
                            state['_param_fp32'] = torch.empty(p.size(), dtype=torch.float32, device=p.device)   # on device
                            state['_param_fp32'].copy_(p)
                            
                        state['neg_pre_grad'] = p.grad.clone().mul_(-1.0)
                
                        

                    # update the steps for each param group update
                    state['step'] += 1
                    
                    bias_correction1 = 1.0 - beta1**group['step']
                    bias_correction2 = 1.0 - beta2**group['step']
                    bias_correction3_sqrt = math.sqrt(1.0 - beta3**group['step'])
                        
                    if p.dtype == torch.half:
                        C.f_adan(
                            state["_param_fp32"],   # fp32
                            p,                      # fp16
                            p.grad,                 # fp16
                            state['neg_pre_grad'],
                            state['exp_avg'],       # fp32: m
                            state['exp_avg_diff'],  # fp32: diff
                            state["exp_avg_sq"],    # fp32: v
                            beta1,
                            beta2,
                            beta3,
                            bias_correction1,
                            bias_correction2,
                            bias_correction3_sqrt,
                            group['lr'],
                            group['weight_decay'],
                            group['eps'],
                            scale
                        )
                    else:
                        _single_tensor_adan(
                            p,
                            p.grad / scale,
                            state['neg_pre_grad'],
                            state['exp_avg'],
                            state['exp_avg_diff'],
                            state["exp_avg_sq"],
                            beta1,
                            beta2,
                            beta3,
                            bias_correction1,
                            bias_correction2,
                            bias_correction3_sqrt,
                            group['lr'],
                            group['weight_decay'],
                            group['eps']
                        )
                        
                    state['neg_pre_grad'].zero_().add_(p.grad, alpha=-1.0)

        return loss