import numpy as np
import torch
import torch.nn as nn
#import utils
from . import utils
from . import L0_Utils


def L0_PGD_AttackWrapper(model, device, dataLoader, n_restarts, num_steps, step_size, sparsity, random_start):

    model.eval()

    if random_start and n_restarts == 1:
            raise ValueError(
            f"Invalid parameter combination: random_start={random_start} and n_restarts={n_restarts}. "
            f"When using multiple restarts (n_restarts > 1), random_start should be False, "
            f"or use n_restarts=1 with random_start=True."
        )
    
    total_batches = len(dataLoader)
    total_samples = len(dataLoader.dataset)
    
    # First pass: Collect all original examples and labels
    x_tensor, y_tensor = utils.DataLoaderToTensor(dataLoader)
    all_original_examples, all_labels = utils.TensorToNumpy(x_tensor, y_tensor)
    
    # Initialize adversarial examples and robust accuracy
    all_adv_examples = np.copy(all_original_examples)
    all_latest_attempts = np.copy(all_original_examples)  # ------> Added to save failed adv examples as well
    pgd_adv_acc = None
    
    # Outer loop: Multiple restarts
    for counter in range(n_restarts):
        print(f"Restart {counter + 1}/{n_restarts}")
        
        if counter == 0:
            # Get initial predictions on clean images
            corr_pred = utils.get_predictions(model, all_original_examples, all_labels, device)
            pgd_adv_acc = np.copy(corr_pred)
        
        # Inner loop: Process each batch in this restart
        batch_start_idx = 0
        for batch_idx, (x_batch, y_batch) in enumerate(dataLoader):
            x_numpy, y_numpy = utils.TensorToNumpy(x_batch, y_batch)
            batch_size = x_numpy.shape[0]
            batch_end_idx = batch_start_idx + batch_size
            
            # Get the original samples for this batch
            x_nat = all_original_examples[batch_start_idx:batch_end_idx]
            y_nat = all_labels[batch_start_idx:batch_end_idx]
            
            # Perform attack on this batch
            x_batch_adv, curr_pgd_adv_acc = L0_Utils.perturb_L0_box(
                model, x_nat, y_nat, 
                -x_nat, 
                1.0 - x_nat, 
                sparsity, num_steps, step_size, device, random_start
            )
            
            # Update robust accuracy: take minimum (worst case) across restarts
            pgd_adv_acc[batch_start_idx:batch_end_idx] = np.minimum(
                pgd_adv_acc[batch_start_idx:batch_end_idx], 
                curr_pgd_adv_acc
            )
            
            # Update adversarial examples for samples that were successfully attacked
            mask = np.logical_not(curr_pgd_adv_acc)
            all_adv_examples[batch_start_idx:batch_end_idx][mask] = x_batch_adv[mask]

            # NEW: Always store latest attempt for ALL samples (will use for failed ones later)    ----> Added later to return failed adversarial
            all_latest_attempts[batch_start_idx:batch_end_idx] = x_batch_adv
            
            batch_start_idx = batch_end_idx

    # NEW: After all restarts, fill in failed samples with their latest attempts
    still_robust = pgd_adv_acc.astype(bool)  # True where attack never succeeded
    all_adv_examples[still_robust] = all_latest_attempts[still_robust]
    
    # Calculate overall statistics
    overall_robust_acc = np.sum(pgd_adv_acc) / total_samples * 100.0

    # Calculate pixels changed statistics
    pixels_changed = np.sum(np.amax(np.abs(all_adv_examples - all_original_examples) > 1e-10, axis=-1), axis=(1,2))

    # Calculate maximum perturbation across all samples
    max_perturbation = np.amax(np.abs(all_adv_examples - all_original_examples))

    # print('Pixels changed: ', pixels_changed)    # Uncomment it to print total pixel changes in each samples
    print(f"{'='*70}")
    print(f"Total samples processed: {total_samples}")
    print(f"Overall Robust Accuracy at {sparsity} pixels:: {overall_robust_acc:.2f}%")
    print(f"Maximum perturbation size: {max_perturbation:.5f}")
    print(f"{'='*70}\n")

    # Print per-class robust accuracy using the new function
    # utils.print_per_class_robust_accuracy(all_labels, all_robust_acc)   # Uncomment to print classwise accuracy
    
    # Convert numpy arrays back to tensors using NumpyToTensor
    xAdv, yClean = utils.NumpyToTensor(all_adv_examples, all_labels)
    
    # Create and return adversarial dataLoader
    advLoader = utils.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)
     
    return advLoader