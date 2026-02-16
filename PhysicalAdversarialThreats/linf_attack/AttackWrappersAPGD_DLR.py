# --- DLR loss (untargeted), binary-safe, per-sample---
import torch 
import DataManagerPytorch as DMP
import torchvision

# project into Linf ball around x_orig  ———
def projection_linf(x_adv, x_orig, eps):
    return torch.max(
        torch.min(x_adv, x_orig + eps),
        x_orig - eps
    )    

# --- DLR (untargeted) sort, NO denominator ---
def dlr_loss(logits, y):

    # sort ASCENDING to get [-1] = max, [-2] = 2nd max
    x_sorted, ind_sorted = logits.sort(dim=1, descending=False)
    # is the top-1 index the true label?
    is_top = (ind_sorted[:, -1] == y).float()
    u = torch.arange(logits.shape[0], device=logits.device)

    zy = logits[u, y]
    # if argmax == y use 2nd max, else use max
    z_secondmax = x_sorted[:, -2]   # second largest
    z_max       = x_sorted[:, -1]   # largest

    losses = -( zy - z_secondmax*is_top - z_max*(1. - is_top) )
    return losses  # [B]


def get_grad_dlr(model, x, y):
    """
    Per-sample grad for dlr_loss_simple (same structure as CE get_grad).
    """
    B = x.size(0)
    grads = torch.zeros_like(x)
    for i in range(B):
        xi = x[i:i+1].detach().clone().requires_grad_(True)
        yi = y[i:i+1].long()
        logits_i = model(xi)
        li = dlr_loss(logits_i, yi).mean()
        gi = torch.autograd.grad(li, xi, retain_graph=False, create_graph=False)[0]
        grads[i] = gi[0].detach()
    return grads


# APGD wrapper, DLR variant (instead of CE) ---
def APGDNativePytorch_DLR(device, dataLoader, model, eps_max, num_steps,
                          alpha=0.75, rho=0.75, clip_min=0, clip_max=1.0,
                          random_start=False, eta_scalar=None):
    model.eval()

    N = len(dataLoader.dataset)
    C, H, W = DMP.GetOutputShape(dataLoader)

    x_adv_all = torch.zeros(N, C, H, W)
    y_all     = torch.zeros(N, dtype=torch.long)

    # checkpoints W
    W = [0]
    p_prev2, p_prev1 = 0.0, 0.22
    W.append(int(p_prev1 * num_steps))
    while W[-1] < num_steps:
        delta  = max(p_prev1 - p_prev2 - 0.03, 0.06)
        p_next = p_prev1 + delta
        w_next = int(p_next * num_steps)
        W.append(w_next)
        p_prev2, p_prev1 = p_prev1, p_next

    if eta_scalar is None:
        eta_scalar = 0.05 * eps_max   # matches current CE default

    idx_out = 0
    for x_clean, y in dataLoader:
        bs = x_clean.size(0)
        x_clean = x_clean.to(device)
        y       = y.to(device).long()

        # x(0)
        if random_start:
            delta = torch.empty_like(x_clean).uniform_(-eps_max, eps_max)
            x_k   = torch.clamp(x_clean + delta, clip_min, clip_max).detach()
        else:
            x_k = x_clean.clone().detach()

        # first step  x(1)
        grad   = get_grad_dlr(model, x_k, y)
        z_next = torch.clamp(x_k + eta_scalar * grad.sign(), clip_min, clip_max)
        x_next = projection_linf(z_next, x_clean, eps_max).detach()

        # (x_max, f_max)
        f_x0 = torch.empty(bs, device=x_clean.device, dtype=torch.float32)
        f_x1 = torch.empty(bs, device=x_clean.device, dtype=torch.float32)
        with torch.no_grad():
            for i in range(bs):
                f_x0[i] = dlr_loss(model(x_clean[i:i+1]), y[i:i+1]).sum()
                f_x1[i] = dlr_loss(model(x_next[i:i+1]),  y[i:i+1]).sum()

        better = (f_x1 > f_x0)
        x_max  = x_clean.clone()
        x_max[better] = x_next[better]
        f_max  = f_x0.clone()
        f_max[better] = f_x1[better]

        # trackers
        x_prev = x_k.clone()
        x_k    = x_next.clone()
        eta = torch.full((bs,1,1,1), eta_scalar, device=device, dtype=x_clean.dtype)
        improvement = torch.zeros(bs, device=device, dtype=torch.int32)
        checkpoint_ptr = 1
        prev_eta   = eta.clone()
        prev_f_max = f_max.clone()

        for k in range(1, num_steps):
            grad   = get_grad_dlr(model, x_k, y)
            z_next = torch.clamp(x_k + eta * grad.sign(), clip_min, clip_max)
            z_next = projection_linf(z_next, x_clean, eps_max)

            x_next = x_k + alpha * (z_next - x_k) + (1 - alpha) * (x_k - x_prev)
            x_next = projection_linf(x_next, x_clean, eps_max)
            x_next = torch.clamp(x_next, clip_min, clip_max).detach()

            f_k    = torch.empty(bs, device=x_clean.device, dtype=torch.float32)
            f_next = torch.empty(bs, device=x_clean.device, dtype=torch.float32)
            with torch.no_grad():
                for i in range(bs):
                    f_k[i]    = dlr_loss(model(x_k[i:i+1]),    y[i:i+1]).sum()
                    f_next[i] = dlr_loss(model(x_next[i:i+1]), y[i:i+1]).sum()

            improvement += (f_next > f_k).to(torch.int32)
            better2 = (f_next > f_max)
            x_max[better2] = x_next[better2]
            f_max[better2] = f_next[better2]

            # checkpoint restart-to-best (fixed logic)
            if checkpoint_ptr < len(W) and k == W[checkpoint_ptr]:

                interval = W[checkpoint_ptr] - W[checkpoint_ptr - 1]
                cond1 = improvement.to(torch.float32) < (rho * interval)
                same_eta_flat = (eta == prev_eta).all(dim=(1,2,3))
                cond2 = same_eta_flat & (f_max == prev_f_max)
                for i in range(bs):
                    if cond1[i].item() or cond2[i].item():
                        eta[i]    = eta[i] / 2.0
                        x_next[i] = x_max[i].clone()
                        x_k[i]    = x_max[i].clone()
                improvement.zero_()
                prev_eta   = eta.clone()
                prev_f_max = f_max.clone()
                checkpoint_ptr += 1

            x_prev = x_k.clone()
            x_k    = x_next.clone()

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
