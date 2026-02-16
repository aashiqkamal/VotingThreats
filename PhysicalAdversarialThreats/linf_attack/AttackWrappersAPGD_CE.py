#Attack wrappers class for APGD 
import torch 
import DataManagerPytorch as DMP
import torchvision

# project into Linf ball around x_orig  ———
def projection_linf(x_adv, x_orig, eps):
    return torch.max(
        torch.min(x_adv, x_orig + eps),
        x_orig - eps
    )    
    
def get_grad(model, x, y, loss_fn): #Return per-sample input gradients 
    
    B = x.size(0) #BS 
    grads = torch.zeros_like(x)   # holder for B persample grads

    for i in range(B): # single sample 
        xi = x[i:i+1].detach().clone().requires_grad_(True)  # xi - ith image, keeps the batch dimension using i+1 slicing, fresh tensor (removed from computational graph),new copy in memory ,dloss/dxi. 
        #yi = y[i:i+1]  # [1], label , take the shape. 
        yi = y[i:i+1].long() # changed this for voting dataset
        
        logits_i = model(xi)
        loss_i   = loss_fn(logits_i, yi)  # shape [1] because reduction= none
        loss_i_s   = loss_i.sum() # turn it into a scalar with one element
        gi = torch.autograd.grad(loss_i_s, xi)[0]  # [1,C,H,W]
        grads[i] = gi[0].detach() # store as [C,H,W]

    return grads

def APGDNativePytorch(device, dataLoader, model, eps_max, num_steps, eta_scalar,
                      alpha=0.75, rho=0.75, clip_min=-1.0, clip_max=1.0,
                      random_start=False):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    N = len(dataLoader.dataset)
    C, H, W = DMP.GetOutputShape(dataLoader)

    x_adv_all = torch.zeros(N, C, H, W)
    y_all     = torch.zeros(N, dtype=torch.long)

    # build checkpoint  W
    W = [0] # set of checkpoint iterations, W is for checkpoint which come up with iteration number where the algorithm will check the two conditions
    p_prev2, p_prev1 = 0.0, 0.22 #   p0=0, p1=0.22,
    W.append(int(p_prev1 * num_steps))
    while W[-1] < num_steps:
        delta  = max(p_prev1 - p_prev2 - 0.03, 0.06) #max(pj-pj-1-0.03, 0.06) 
        p_next = p_prev1 + delta #pj+1 = pj + max(pj-pj-1-0.03, 0.06) 
        w_next = int(p_next * num_steps) #wj=pj*N_iter
        W.append(w_next)
        p_prev2, p_prev1 = p_prev1, p_next

    idx_out = 0
    
    # Loop over samples
    for x_clean, y in dataLoader:
        bs = x_clean.size(0)
        x_clean = x_clean.to(device)
        #y       = y.to(device)
        y       = y.to(device).long() # changed this for voting dataset. 
        

        # x(0) random start
        if random_start:
            delta = torch.empty_like(x_clean).uniform_(-eps_max, eps_max)
            x_k   = torch.clamp(x_clean + delta, clip_min, clip_max).detach()
        else:
            x_k = x_clean.clone().detach()

        # line 3: first step with scalar eta , First step --> x(1)
        #eta_scalar = 0.1 * eps_max  #changed for voting dataset
        grad   = get_grad(model, x_k, y, loss_fn)  # (bs,C,H,W), per-sample loop inside, x_k loss
        z_next = torch.clamp(x_k + eta_scalar * grad.sign(), clip_min, clip_max) #valid image range within clip_min and max. 
        x_next = projection_linf(z_next, x_clean, eps_max).detach() # apply PS(.)

        # lines 4,5: pick (x_max, f_max) per-sample
        f_x0 = torch.empty(bs, device=x_clean.device, dtype=torch.float32)
        f_x1 = torch.empty(bs, device=x_clean.device, dtype=torch.float32)
        
        with torch.no_grad():
            for i in range(bs):
                f_x0[i] = loss_fn(model(x_clean[i:i+1]), y[i:i+1]).sum() #f_x1 is adv loss, f_x0 is clean image loss, converting scalar
                f_x1[i] = loss_fn(model(x_next[i:i+1]),  y[i:i+1]).sum()

        better = (f_x1 > f_x0)        # [bs] bool [true, false,], which indices improved , 
        x_max  = x_clean.clone()
        x_max[better] = x_next[better] # pick the better image , x_max = [clean0, clean1], better= [T, F], x_max[better]= clean0, x_next[better]= adv0, so adv0 selected.and clean1 will not replace
        f_max  = f_x0.clone()
        f_max[better] = f_x1[better] # pick the corresponding higher loss of adv0 and clean1. 

        # Initialize per-sample trackers for main loop 
        x_prev = x_k.clone() #x_0 (either clean or random) 
        x_k    = x_next.clone() #x_1 (first aggre step of attack) 

        # start per-sample eta from the scalar used at line 3 (added in this version)
        eta = torch.full((bs,1,1,1), eta_scalar, device=device, dtype=x_clean.dtype) # new tensor, shape = grad [bs,C,H,W], one value per sample filling with 2 * eps_max, 

        # per-sample improvement counters (since last checkpoint) 
        improvement = torch.zeros(bs, device=device, dtype=torch.int32) #1D tensor as BS. 

        checkpoint_ptr = 1   # W[0]==0, we check at k+1 == W[1], …
        prev_eta   = eta.clone()
        prev_f_max = f_max.clone()
        

        # Lines 6–17 
        for k in range(1, num_steps):
            # z(k+1)
            grad   = get_grad(model, x_k, y, loss_fn)    # (bs,C,H,W) per-sample
            z_next = torch.clamp(x_k + eta * grad.sign(), clip_min, clip_max) # 
            z_next = projection_linf(z_next, x_clean, eps_max)

            # x(k+1) with momentum
            x_next = x_k + alpha * (z_next - x_k) + (1 - alpha) * (x_k - x_prev)
            x_next = projection_linf(x_next, x_clean, eps_max)
            x_next = torch.clamp(x_next, clip_min, clip_max).detach()

            # per-sample : one losses per image  
            f_k    = torch.empty(bs, device=x_clean.device, dtype=torch.float32) # initially empty 
            f_next = torch.empty(bs, device=x_clean.device, dtype=torch.float32)

            with torch.no_grad():
                for i in range(bs): # loop over bacth size 
                    f_k[i]    = loss_fn(model(x_k[i:i+1]),y[i:i+1]).sum()  # 
                    f_next[i] = loss_fn(model(x_next[i:i+1]), y[i:i+1]).sum()           


           
            improvement += (f_next > f_k).to(torch.int32)  # count improvements since last checkpoint, check/count how many times f_next is greater than f_k 

            # update (x_max, f_max) 
            better2 = (f_next > f_max) # [bs] bool [true, false, true, ...]
            x_max[better2] = x_next[better2]
            f_max[better2] = f_next[better2]

            # checkpoint: bs=1 
            if (k  == W[checkpoint_ptr] and checkpoint_ptr < len(W)) :
                interval = W[checkpoint_ptr] - W[checkpoint_ptr - 1]  # steps since last check

                # Condition 1: per-sample fraction of successes < rho
                cond1 = improvement.to(torch.float32) < (rho * interval)  # The maximum possible number of successes is interval, if this success rate is below rho then we consider progress insufficient, halve the step size and restart from the best point

                # Condition 2: per-sample (eta unchanged) AND (f_max unchanged)
                same_eta_flat = (eta == prev_eta).all(dim=(1, 2, 3))  # [bs] element wise comparison True/False per sample,true: eta unchanged or equal for this sample, 4D->1D  
                
                cond2 = same_eta_flat & (f_max == prev_f_max)          #  [bs] best loss not changed


                for i in range(bs):
                    if cond1[i].item() or cond2[i].item():
                        eta[i]    = eta[i] / 2.0    # each sample i uses its own scalar eta[i]
                        x_next[i] = x_max[i].clone()  # Override tentative with best (new adjustment)
                        x_k[i]    = x_max[i].clone() # removed at new version 
                        #x_prev[i] = x_max[i].clone()  # removed at new version

                # reset for next interval, snapshot current states
                improvement.zero_()
                prev_eta   = eta.clone()
                prev_f_max = f_max.clone()
                checkpoint_ptr += 1

            # shift for next iter
            x_prev = x_k.clone()
            x_k    = x_next.clone()
        # ============== end main loop ==============

        # save batch
        x_adv_all[idx_out:idx_out+bs] = x_max.cpu()
        y_all    [idx_out:idx_out+bs] = y.cpu()
        idx_out += bs

        torch.cuda.empty_cache()

    advLoader = DMP.TensorToDataLoader(
        x_adv_all, y_all,
        transforms=None,
        batchSize=dataLoader.batch_size,
        randomizer=None
    )
    return advLoader

