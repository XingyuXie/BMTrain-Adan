import torch
from torch import Tensor
import math
from . import _cuda_adan2nd as C

import bmtrain.distributed as dist
from ..global_var import config

def hutchinson(loss_small, p, scale):
    # Compute the gradient of the loss
    grad = torch.autograd.grad(loss_small, p, create_graph=True)[0]

    # Create a random vector u of the same shape as grad
    u = torch.randn_like(grad).div_(math.sqrt(scale))

    # Compute the dot product of grad and u
    grad_dot_u = torch.dot(grad.view(-1), u.view(-1))

    # Compute the Hessian-vector product of grad_dot_u and p
    hessian_vector_product = torch.autograd.grad(grad_dot_u, p, retain_graph=True)[0]

    # Compute the Hutchinson estimation
    hutchinson_estimation = u * hessian_vector_product

    # All-reduce to ensure consistency across nodes
    dist.all_reduce(hutchinson_estimation)

    # Divide by the number of nodes to get the average
    world_size = config[world_size]
    hutchinson_estimation /= world_size

    return hutchinson_estimation # the real hessian-vector product * scale


def _single_tensor_adan(
    param: Tensor,
    grad: Tensor,
    neg_pre_grad: Tensor,
    # hessian_estimate: Tensor,
    exp_avg: Tensor,
    exp_avg_diff: Tensor,
    h_sq: Tensor,
    *,
    beta1: float,
    beta2: float,
    # beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3: float,
    lr: float,
    weight_decay: float,
    eps: float,
    rho_bs=256.0,
    scale
):
    grad.div_(scale)
    
    # for memory saving, we use `neg_grad_or_diff`
    # to get some temp variable in a inplace way
    neg_grad_or_diff = neg_pre_grad
    neg_grad_or_diff.add_(grad)

    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
    exp_avg_diff.mul_(beta2).add_(neg_grad_or_diff,
                                    alpha=1 - beta2)  # diff_t

    # # neg_grad_or_diff.mul_(beta2).add_(grad)
    # if hessian_estimate is not None:
    #     hessian_estimate.div_(scale)
    #     h_sq.mul_(beta3).add_(1 - beta3, hessian_estimate)
    
    denom = (h_sq / bias_correction3).mul_(rho_bs).add_(eps)
    torch.div(exp_avg, bias_correction1, out=neg_grad_or_diff)
    torch.add(neg_grad_or_diff, exp_avg_diff,
              alpha=beta2 / bias_correction2, out=neg_grad_or_diff)
    neg_grad_or_diff.div_(denom).clamp_(min=-1.0, max=1.0)
    # step_size_diff = lr * beta2 / bias_correction2
    # step_size = lr / bias_correction1
    # param.addcdiv_(exp_avg, denom, value=-step_size)
    # param.addcdiv_(exp_avg_diff, denom, value=-step_size_diff)
    param.add_(neg_grad_or_diff, alpha=-lr)
    param.div_(1 + lr * weight_decay)



class Adan2ndOptimizer(torch.optim.Optimizer):
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
                 eps=1e-15,
                 weight_decay=0.0,
                 rho=0.03,
                 bs=256):
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
                        weight_decay=weight_decay,
                        rho=rho,
                        bs=bs)
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
                        state['h_sq'] *= delta
                        state['neg_pre_grad'] *= delta
                        
    def update_hessian(self, scale=1.0):
        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            if 'h_step' in group:
                group['h_step'] += 1
            else:
                group['h_step'] = 1
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'h_sq' not in state:
                    state['h_sq'] = torch.zeros(p.size(), dtype=torch.float32, device=p.device)   # on device
                if p.dtype == torch.half:
                    state['h_sq'].mul_(beta3).addcmul_(p.grad.float(), p.grad.float(), value=(1 - beta3)/scale)
                else:
                    state['h_sq'].mul_(beta3).addcmul_(p.grad, p.grad, value=(1 - beta3)/scale)
                # h_sq = real * scale


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
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

                
            # update the steps for each param group update
            bias_correction1 = 1.0 - beta1**group['step']
            bias_correction2 = 1.0 - beta2**group['step']
            bias_correction3 = 1.0 - beta3**group.get('h_step',1)

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
                        state['h_sq'] = torch.zeros(p.size(), dtype=torch.float32, device=p.device)   # on device

                        if p.dtype == torch.half:
                            state['_param_fp32'] = torch.empty(p.size(), dtype=torch.float32, device=p.device)   # on device
                            state['_param_fp32'].copy_(p)
                            
                        state['neg_pre_grad'] = p.grad.clone().mul_(-1.0)
                        
                    # hessian_estimate=None
                    # if loss_small is not None:
                    #     hessian_estimate = hutchinson(loss_small, p, scale)
                        
                    if p.dtype == torch.half:
                        C.f_adan2nd(
                            state["_param_fp32"],   # fp32
                            p,                      # fp16
                            p.grad,                 # fp16
                            state['neg_pre_grad'],  # fp16
                            # hessian_estimate,       # fp16
                            state['exp_avg'],       # fp32: m
                            state['exp_avg_diff'],  # fp32: diff
                            state["h_sq"],    # fp32: v
                            beta1,
                            beta2,
                            # beta3,
                            bias_correction1,
                            bias_correction2,
                            bias_correction3,
                            group['lr'],
                            group['weight_decay'],
                            group['eps'],
                            group['rho']*group['bs'],
                            scale
                        )
                    else:
                        _single_tensor_adan(
                            p,
                            p.grad,
                            state['neg_pre_grad'],
                            # hessian_estimate,
                            state['exp_avg'],
                            state['exp_avg_diff'],
                            state["h_sq"],
                            beta1,
                            beta2,
                            # beta3,
                            bias_correction1,
                            bias_correction2,
                            bias_correction3,
                            group['lr'],
                            group['weight_decay'],
                            group['eps'],
                            group['rho']*group['bs'],
                            scale
                        )
                    state['neg_pre_grad'].zero_().add_(p.grad, alpha=-1.0)
        return loss