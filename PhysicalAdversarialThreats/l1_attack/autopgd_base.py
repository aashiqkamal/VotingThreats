## L1 separated version from original AutoAttack

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .other_utils import L0_norm, L1_norm
from .checks import check_zero_gradients


def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''
    x = x2.clone().float().view(x2.shape[0], -1) #original image x2 to 1D
    y = y2.clone().float().view(y2.shape[0], -1) #current perturbation y2 to 1D , perturbation vector d =u-x 
    sigma = y.clone().sign() # sign (=/-)of y vector s
    u = torch.min(1 - x - y, x + y) # compute y_i (gamma),   u=x+y? 
    u = torch.min(torch.zeros_like(y), u) #-u is exactly min(|u_i-x_i|, y_i)
    l = -torch.clone(y).abs() # -|y| corresponds to the point where the shrink lamda equals the perturbation magnitude |di|. #negative absolute current perturbation
    d = u.clone()
    #line 7
    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1) # concatenates all breakpoints (u and l) into a single long vector and sorts them (bs). indbs are their indices
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1) #bs2 is bs shifted left with a zero appended
 
    inu = 2*(indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1) #pre-calculates the slope of the piecewise linear function between each of the sorted breakpoints.

    #checking of easy case
    s1 = -u.sum(dim=1 # line 5

    c = eps1 - y.clone().abs().sum(dim=1) #remaining budget when starting from current perturbation y, c represents how much extra L1 we can still add to |y|_1 before hitting eps
    c5 = s1 + c < 0   #checks if this is over the budget (c5), S > eps (line 6), if exceeds budget need lamda_star
    c2 = c5.nonzero().squeeze(1) #check for hard case? 

    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1) #This vector pre-computes the value of the function f(lamda) at every breakpoint. S array is what will compare to eps to find where S crosses eps.

    if c2.nelement != 0:   # The if S > eps check is equivalent to it

      lb = torch.zeros_like(c2).float()
      ub = torch.ones_like(lb) *(bs.shape[1] - 1)

      nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
      counter2 = torch.zeros_like(lb).long()
      counter = 0
       
      #implemented by concatenating breakpoints u and l, sorting them, building cumulative arrays, 
      #then binary-searching (lb/ub) to find the interval where the sum crosses eps, followed by computing alpha = lamda_star.
      while counter < nitermax:  # binary search to find the lamda_star  line 10-29 
        counter4 = torch.floor((lb + ub) / 2.)  # dividing the middle
        counter2 = counter4.type(torch.LongTensor)

        c8 = s[c2, counter2] + c[c2] < 0   #The c8 = s ... < 0 is the check that decides which half of the search space to discard.
        ind3 = c8.nonzero().squeeze(1)     #logic to update lb, ub 
        ind32 = (~c8).nonzero().squeeze(1)
        if ind3.nelement != 0:
            lb[ind3] = counter4[ind3]
        if ind32.nelement != 0:
            ub[ind32] = counter4[ind32]

        counter += 1

      lb2 = lb.long()  
      alpha = (-s[c2, lb2] -c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]  # Once the binary search finds the correct interval, it finds the exact value of lambda star using linear interpolation.
      d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2]) # This line uses the alpha to compute the final, corrected magnitudes for the perturbation for each pixel. 

    return (sigma * d).view(x2.shape)     #uses alpha and clamps to yield final d, which equals zi in magnitude.


class APGDAttack():
    """
    AutoPGD (L1-only, CE & DLR)
    https://arxiv.org/abs/2003.01690

    Kept function & method names/signatures identical to the upstream file,
    but restricted to L1 norm and untargeted CE / DLR losses.
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='L1',
            n_restarts=1,
            eps=None,
            seed=0,
            loss='ce',
            eot_iter=1,
            rho=.75,
            topk=None,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None):
        """
        AutoPGD implementation in PyTorch (L1 + CE/DLR only).
        Names/args kept to preserve external API compatibility.
        """
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        self.use_largereps = use_largereps
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0. if eps is not None else None
        self.is_tf_model = is_tf_model  # kept
        self.y_target = None
        self.logger = logger

        # L1 only
        assert self.norm in ['L1'], 'This minimal build supports only L1.'
        assert self.eps is not None
        if self.is_tf_model:
            raise ValueError('TF models are not supported in this minimal L1 build.')

        # parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

    def init_hyperparam(self, x):
        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        # L1 normalize to avoid changing names
        try:
            t = x.abs().view(x.shape[0], -1).sum(dim=-1)
        except:
            t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)
        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

   # def dlr_loss(self, x, y):
        #x_sorted, ind_sorted = x.sort(dim=1)
        #ind = (ind_sorted[:, -1] == y).float()
        #u = torch.arange(x.shape[0], device=x.device)
    def dlr_loss(self, x, y):
        """
        DLR loss with binary-safe 
        - C >= 3 (original AutoAttack):  -(z_y - z_k) / (z_(1) - z_(3) + eps)
        - C = 2 (binary):                -(z_y - z_k) 
        Returns: [B] tensor
        """
        x_sorted, ind_sorted = x.sort(dim=1)     # ascending: [:, -1]=max=z_(1), [:, -2]=z_(2), [:, -3]=z_(3)
        B, C = x.shape
        u = torch.arange(B, device=x.device)

        if C >= 3:
            # --- ORIGINAL (unchanged) ---
            ind = (ind_sorted[:, -1] == y).float()  # is y the argmax?
            return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

        elif C == 2:
            # --- BINARY-SAFE --- aligned with paper
            # y must be 0/1; pick the other class logit directly
            other = y ^ 1  # same as (1 - y), but explicit for {0,1}
            zy = x[u, y]
            zk = x[u, other]
            return -(zy - zk)  
            #zmax = x_sorted[:, -1]                      # max logit
            #zmin = x_sorted[:,  0]                      # min logit
            #is_top = (ind_sorted[:, -1] == y)           # y is argmax?
            #zk = torch.where(is_top, zmin, zmax)        # second max if y is max; else max
            #zy = x[u, y]
            #return -(zy - zk) / (zmax - zmin + 1e-12)

        else:
            # C == 1 (degenerate): no meaningful margin
            return torch.zeros(B, device=x.device, dtype=x.dtype)




    def attack_single_run(self, x, y, x_init=None):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            
        if x_init is not None:
            x_adv = x_init.clone()
            if self.verbose:
                print('[custom init] L1 perturbation {:.5f}'.format(
                    (x_adv - x).abs().view(x.shape[0], -1).sum(1).max()))
        
        elif getattr(self, "use_rs", True):
        # random L1 init (projection onto L1-ball  [0,1])
            t = torch.randn(x.shape, device=self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = (x + t + delta)
        else:
        # deterministic start: exactly the clean image
            x_adv = x.clone()        

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]], device=self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]], device=self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        # CE / DLR only (PyTorch)
        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduction='none')
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError('unknown loss (allowed: ce, dlr)')

        #  gradient at init  line 3–4: compute VL(x(i)) 
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()
        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        if self.loss in ['dlr']:
            check_zero_gradients(grad, logger=self.logger)

        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        alpha = 1.0  # L1 uses alpha=1 in upstream
        #alpha = 0.05 # for voting dataset some cases
        step_size = alpha * self.eps * torch.ones([x.shape[0], *([1] * self.ndims)], device=self.device).detach()
        x_adv_old = x_adv.clone()
        k = max(int(.04 * self.n_iter), 1)
        n_fts = math.prod(self.orig_dim)

        # init sparsity schedule
        if x_init is None:
            topk = .2 * torch.ones([x.shape[0]], device=self.device)
            sp_old = n_fts * torch.ones_like(topk)
        else:
            topk = L0_norm(x_adv - x) / n_fts / 1.5
            sp_old = L0_norm(x_adv - x)

        adasp_redstep = 1.5
        adasp_minstep = 10.
        counter3 = 0
        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)

        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter): #Loop here line 4
            # gradient step (L1 sparse update + projection)
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0] #line 6 (compute sparse sign of gradient with k.d active coordinates) 
                topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                grad_topk = grad_topk[u, topk_curr].view(-1, *[1] * (len(x.shape) - 1))
                sparsegrad = grad * (grad.abs() >= grad_topk).float()
                x_adv_1 = x_adv + step_size * sparsegrad.sign() / (L1_norm(sparsegrad.sign(), keepdim=True) + 1e-10) #line 14 step to u, sparsegrad.sign() --S(VL,k.d) 

                delta_u = x_adv_1 - x  #line 15 -- proposed perturbation relative to clean x
                delta_p = L1_projection(x, delta_u, self.eps)
                #x_adv = (x + delta_u + delta_p)
                x_adv = (x_adv_1 + delta_p)
                #x_adv   = x + delta_p.clamp(0., 1.)
            # get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            ind_pred = (pred == 0).nonzero().squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
            if self.verbose:
                str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(step_size.mean(), topk.mean() * n_fts)
                print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
                    i, loss_best.sum(), acc.float().mean(), str_stats))

            # step-size & sparsity adaptation (L1)
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1 + 0
                ind = (y1 > loss_best).nonzero().squeeze() #track best loss/point 
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1
                if counter3 == k:  
                    sp_curr = L0_norm(x_best - x)  #  line 7 -- update k 
                    fl_redtopk = (sp_curr / sp_old) < .95 # did the sparsity change enough? 
                    topk = sp_curr / n_fts / 1.5   #equation 9 
                    step_size[fl_redtopk] = alpha * self.eps   # line 8 -- update n eta 
                    step_size[~fl_redtopk] /= adasp_redstep    # reduce as per equation 10 
                    step_size.clamp_(alpha * self.eps / adasp_minstep, alpha * self.eps)  #keep within eps and eps/10 
                    sp_old = sp_curr.clone()

                    x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                    grad[fl_redtopk] = grad_best[fl_redtopk].clone()

                    counter3 = 0

        return (x_best, acc, loss_best, x_best_adv)

    def perturb(self, x, y=None, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True return points with highest loss (not used by wrapper)
        :param x_init:      optional custom initialization
        """
        assert self.loss in ['ce', 'dlr']
        if y is not None and len(y.shape) == 0:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        y_pred = self.model(x).max(1)[1]
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)  # not taken, y passed
        else:
            y = y.detach().clone().long().to(self.device)  #it is executed

        adv = x.clone()  # fill with adversarials
        acc = (y_pred == y)  # starts with correctly classified in this batch
        loss = -1e10 * torch.ones_like(acc).float()

        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        if self.use_largereps: # not executed, default is false
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig, .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1])

        if not best_loss: # True, as best loss pass yet (first run)
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            for counter in range(self.n_restarts): # n_restart= 1 or 5 any value
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()

                    if not self.use_largereps: # True (as use_largereps is default False)
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool, x_init=x_init) #x_init = NONE here, it calls attack_single_run() the main algorithm 1
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)  # not executed
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr #do
                    ind_curr = (acc_curr == 0).nonzero().squeeze()  #do

                    acc[ind_to_fool[ind_curr]] = 0  # write back fooled examples to the correct original indices 
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()))

            return adv # return this adv
        else: # not executed now, but executed when it is best loss
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]], device=self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y, x_init=x_init)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))
            return adv_best

    def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):
        assert len(epss) == len(iters)
        assert self.norm in ['L1']
        self.use_rs = False
        if not use_rs:
            x_init = None
        else:
            x_init = x + torch.randn_like(x)
            x_init += L1_projection(x, x_init - x, float(epss[0]))
        if self.verbose:
            print('total iter: {}'.format(sum(iters)))
        for eps, niter in zip(epss, iters):
            if self.verbose:
                print('using eps: {:.2f}'.format(eps))
            self.n_iter = niter + 0
            self.eps = eps + 0.
            if x_init is not None:
                x_init += L1_projection(x, x_init - x, float(eps))
            x_init, acc, loss, x_adv = self.attack_single_run(x, y, x_init=x_init)
        return (x_init, acc, loss, x_adv)


# Note: Linf/L2 branches have been removed intentionally
# to keep this file focused on L1 + CE/DLR while preserving the API functional.
