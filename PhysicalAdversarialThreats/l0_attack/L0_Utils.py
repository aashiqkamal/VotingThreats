import numpy as np
import torch
#import utils
from . import utils
from . import L0_Utils

def perturb_L0_box(model, x_nat, y_nat, lb, ub, sparsity, num_steps, step_size, device, random_start):
  if random_start == True:   
    x2 = x_nat + np.random.uniform(lb, ub, x_nat.shape)
    x2 = np.clip(x2, 0, 1)
  else:
    x2 = np.copy(x_nat)
      
  adv_not_found = np.ones(y_nat.shape)
  adv = np.zeros(x_nat.shape)

  for i in range(num_steps):
    if i > 0:
      pred, grad = utils.get_predictions_and_gradients(model, x2, y_nat, device)
      adv_not_found = np.minimum(adv_not_found, pred.astype(int))
      adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])
      grad /= (1e-10 + np.sum(np.abs(grad), axis=(1,2,3), keepdims=True))
      x2 = np.add(x2, (np.random.random_sample(grad.shape)-0.5)*1e-12 + step_size * grad, casting='unsafe')
    x2 = x_nat + project_L0_box(x2 - x_nat, sparsity, lb, ub)

  # Fill in failed cases with final x2
  adv[adv_not_found.astype(bool)] = x2[adv_not_found.astype(bool)]    # ----> Added to return failed adversarial samples as well
    
  return adv, adv_not_found

def project_L0_box(y, k, lb, ub):
  x = np.copy(y)
  p1 = np.sum(x**2, axis=-1)
  p2 = np.minimum(np.minimum(ub - x, x - lb), 0)
  p2 = np.sum(p2**2, axis=-1)
  p3 = np.sort(np.reshape(p1-p2, [p2.shape[0],-1]))[:,-k]
  x = x*(np.logical_and(lb <=x, x <= ub)) + lb*(lb > x) + ub*(x > ub)
  x *= np.expand_dims((p1 - p2) >= p3.reshape([-1, 1, 1]), -1)
  return x

def sigma_map(x):
    ''' creates the sigma-map for the batch x '''
    sh = [4]
    sh.extend(x.shape)
    t = np.zeros(sh)
    t[0,:,:-1] = x[:,1:]
    t[0,:,-1] = x[:,-1]
    t[1,:,1:] = x[:,:-1]
    t[1,:,0] = x[:,0]
    t[2,:,:,:-1] = x[:,:,1:]
    t[2,:,:,-1] = x[:,:,-1]
    t[3,:,:,1:] = x[:,:,:-1]
    t[3,:,:,0] = x[:,:,0]

    mean1 = (t[0] + x + t[1]) / 3
    sd1 = np.sqrt(((t[0] - mean1) ** 2 + (x - mean1) ** 2 + (t[1] - mean1) ** 2) / 3)

    mean2 = (t[2] + x + t[3]) / 3
    sd2 = np.sqrt(((t[2] - mean2) ** 2 + (x - mean2) ** 2 + (t[3] - mean2) ** 2) / 3)

    sd = np.minimum(sd1, sd2)
    sd = np.sqrt(sd)

    return sd

def perturb_L0_sigma(model, x_nat, y_nat, sparsity, num_steps, step_size, device, sigma, kappa, random_start=True):
    if random_start == True:
        x2 = x_nat + np.random.uniform(-kappa, kappa, x_nat.shape)
        x2 = np.clip(x2, 0, 1)
    else:
        x2 = np.copy(x_nat)
    adv_not_found = np.ones(y_nat.shape)
    adv = np.zeros(x_nat.shape)

    for i in range(num_steps):
        if i > 0:
            pred, grad = utils.get_predictions_and_gradients(model, x2, y_nat, device)
            adv_not_found = np.minimum(adv_not_found, pred.astype(int))
            adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])

            grad /= (1e-10 + np.sum(np.abs(grad), axis=(1,2,3), keepdims=True))
            x2 = np.add(x2, (np.random.random_sample(grad.shape) - 0.5) * 1e-12 + step_size * grad, casting='unsafe')

        x2 = project_L0_sigma(x2, sparsity, sigma, kappa, x_nat)

    # Fill in failed cases with final x2
    adv[adv_not_found.astype(bool)] = x2[adv_not_found.astype(bool)]    # Added to return failed adversarial samples as well

    return adv, adv_not_found

def project_L0_sigma(y, k, sigma, kappa, x_nat):
    x = np.copy(y)
    p1 = 1.0 / np.maximum(1e-12, sigma) * (x_nat > 0).astype(float) + 1e12 * (x_nat == 0).astype(float)
    p2 = 1.0 / np.maximum(1e-12, sigma) * (1.0 / np.maximum(1e-12, x_nat) - 1) * (x_nat > 0).astype(float) + \
         1e12 * (x_nat == 0).astype(float) + 1e12 * (sigma == 0).astype(float)
    lmbd_l = np.maximum(-kappa, np.amax(-p1, axis=-1, keepdims=True))
    lmbd_u = np.minimum(kappa, np.amin(p2, axis=-1, keepdims=True))
    
    lmbd_unconstr = np.sum((y - x_nat) * sigma * x_nat, axis=-1, keepdims=True) / \
                    np.maximum(1e-12, np.sum((sigma * x_nat) ** 2, axis=-1, keepdims=True))
    lmbd = np.maximum(lmbd_l, np.minimum(lmbd_unconstr, lmbd_u))
    
    p12 = np.sum((y - x_nat) ** 2, axis=-1, keepdims=True)
    p22 = np.sum((y - (1 + lmbd * sigma) * x_nat) ** 2, axis=-1, keepdims=True)
    p3 = np.sort(np.reshape(p12 - p22, [x.shape[0], -1]))[:, -k]
    
    x = x_nat + lmbd * sigma * x_nat * ((p12 - p22) >= p3.reshape([-1, 1, 1, 1]))
    
    return x