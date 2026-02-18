# AttackMethods.py
# One function per attack (uncomment and run from main file)

import os, sys
import torch
from torch.utils.data import TensorDataset, DataLoader

# -------------------------------------------------
# Make imports work regardless of where user runs from
# -------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(ROOT, "Models"))
sys.path.insert(0, os.path.join(ROOT, "linf_attack"))
sys.path.insert(0, os.path.join(ROOT, "l2_attack"))
sys.path.insert(0, os.path.join(ROOT, "l0_attack"))
sys.path.insert(0, os.path.join(ROOT, "l1_attack"))

import ResNet
import DataManagerPytorch as DMP


# =========================================================
# Model load and dataset handling
# =========================================================
def Load_ResNet20():
    # -----------------------------
    # Choose dataset + checkpoint
    # -----------------------------
    valBlob  = "PATH/TO/val_OnlyBubbles_Grayscale.pth" ## dataset link is provided on readme file
    # valBlob  = "PATH/TO/k_final_dataset_val_Combined_Grayscale.pth"

    modelDir = "PATH/TO/ModelResNet20-VotingCombined-v2-Grayscale-Run1.th"  ## trained model ckpt file drive link is provided on readme file
    # modelDir = "PATH/TO/ModelResNet20-VotingOnlyBubbles-v2-Grayscale-Run1.th"

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------
    # ResNet20 model
    # -----------------------------
    numClasses = 2
    inputImageSize = [1, 1, 40, 50]
    dropOutRate = 0.0

    model = ResNet.resnet20(inputImageSize, dropOutRate, numClasses).to(device)

    checkpoint = torch.load(modelDir, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # -----------------------------
    # Load validation dataset
    # -----------------------------
    data = torch.load(valBlob, weights_only=False)
    images = data["data"].float()                 # [N,1,40,50] in [0,1]
    labels_binary = data["binary_labels"].long()  # [N]

    dataset = TensorDataset(images, labels_binary)
    valLoader = DataLoader(dataset, batch_size=32, shuffle=False)

    cleanAcc = DMP.validateD(valLoader, model, device)
    print("Clean Accuracy (full val):", cleanAcc)

    # -----------------------------
    # Build balanced correctly-classified 1000-sample loader
    # -----------------------------
    totalSamplesRequired = 1000
    correctLoader = DMP.GetCorrectlyIdentifiedSamplesBalanced(
        model, totalSamplesRequired, valLoader, numClasses
    )

    correctAcc = DMP.validateD(correctLoader, model, device)
    print("Clean Accuracy (correctLoader):", correctAcc)
    print("Num samples in correctLoader:", len(correctLoader.dataset))

    return device, model, correctLoader


# =========================================================
# Attack 1: APGD-L2 on ResNet20-C model
# =========================================================
def APGD_L2_ResNet20():
    device, model, correctLoader = Load_ResNet20()

    import autopgd_base  # from l2_attack/autopgd_base.py

    num_steps = 150
    eps_max = 45  # highest value for our dataset images 40x50 
    loss_name = "dlr"

    print("\n--- Running APGD-L2 ---")
    print(f"num_steps={num_steps}, eps_max={eps_max}, loss={loss_name}")

    attackObject = autopgd_base.APGDAttack(
        predict=model,
        n_iter=num_steps,
        norm="L2",
        n_restarts=0,
        eps=eps_max,
        seed=0,
        loss=loss_name,
        eot_iter=1,
        rho=0.75,
        topk=None,
        verbose=False,
        device=device,
        use_largereps=False,
        is_tf_model=False,
        logger=None
    )

    # IMPORTANT: init_hyperparam must be called once
    dummy_x = torch.zeros((1, 1, 40, 50), device=device)
    attackObject.init_hyperparam(dummy_x)

    advLoader = attackObject.APGDCroceAttackWrapper(device, correctLoader)

    advAcc = DMP.validateD(advLoader, model, device)
    print("\nAdversarial Accuracy (APGD-L2):", advAcc)


# =========================================================
# Attack 2: APGD-Linf on ResNet20-C model
# =========================================================
def APGD_Linf_ResNet20():
    device, model, correctLoader = Load_ResNet20()

    from AttackWrappersAPGD_DLR_batch import APGDNativePytorch_DLR as APGD_dlr

    eps = 8/255
    clipMin, clipMax = 0.0, 1.0
    numSteps = 500
    eta_scalar = 2.0 * eps

    print("\n--- Running APGD-Linf (DLR) ---")
    advLoader = APGD_dlr(
        device=device,
        dataLoader=correctLoader,
        model=model,
        eps_max=eps,
        num_steps=numSteps,
        eta_scalar=eta_scalar,
        clip_min=clipMin,
        clip_max=clipMax,
        random_start=False
    )

    advAcc = DMP.validateD(advLoader, model, device)
    print("Adversarial Accuracy (APGD-Linf DLR):", advAcc)


# =========================================================
# Attack 3: L0-PGD on ResNet20-C model
# =========================================================
def L0_PGD_ResNet20():
    device, model, correctLoader = Load_ResNet20()

    from l0_attack import L0_PGD_AttackWrapper as L0_PGD

    n_restarts   = 10
    num_steps    = 30
    step_size    = 15
    sparsity     = 20
    random_start = True

    print("\n--- Running L0_PGD ---")
    print(f"n_restarts={n_restarts}, num_steps={num_steps}, step_size={step_size}, sparsity(k)={sparsity}, random_start={random_start}")

    advLoader = L0_PGD(
        model=model,
        device=device,
        dataLoader=correctLoader,
        n_restarts=n_restarts,
        num_steps=num_steps,
        step_size=step_size,
        sparsity=sparsity,
        random_start=random_start
    )

    advAcc = DMP.validateD(advLoader, model, device)
    print("Adversarial Accuracy (L0_PGD):", advAcc)


# =========================================================
# Attack 4: L0 + Sigma-map PGD on ResNet20-C model
# =========================================================
def L0_Sigma_PGD_ResNet20():
    device, model, correctLoader = Load_ResNet20()

    from l0_attack import L0_Sigma_PGD_AttackWrapper as L0_Sigma

    n_restarts   = 10
    num_steps    = 75
    step_size    = 15
    sparsity     = 2000
    kappa        = 10
    random_start = True

    print("\n--- Running L0_Sigma ---")
    print(f"n_restarts={n_restarts}, num_steps={num_steps}, step_size={step_size}, sparsity(k)={sparsity}, kappa={kappa}, random_start={random_start}")

    advLoader = L0_Sigma(
        model=model,
        device=device,
        dataLoader=correctLoader,
        n_restarts=n_restarts,
        num_steps=num_steps,
        step_size=step_size,
        sparsity=sparsity,
        kappa=kappa,
        random_start=random_start
    )

    advAcc = DMP.validateD(advLoader, model, device)
    print("Adversarial Accuracy (L0_Sigma):", advAcc)


# =========================================================
# Attack 5: L0 + Linf PGD on ResNet20-C model
# =========================================================
def L0_Linf_PGD_ResNet20():
    device, model, correctLoader = Load_ResNet20()

    from l0_attack import L0_Linf_PGD_AttackWrapper as L0_PGD_linf

    eps = 8/255
    k            = 20
    random_start = True
    n_restarts   = 10
    step_size    = 15
    num_steps    = 30

    print("\n--- Running L0 + Linf PGD ---")
    print(f"eps={eps}, k={k}, n_restarts={n_restarts}, num_steps={num_steps}, step_size={step_size}, random_start={random_start}")

    advLoader = L0_PGD_linf(
            model=model,
            device=device,
            dataLoader=correctLoader,
            n_restarts=n_restarts,
            num_steps=num_steps,
            step_size=step_size,
            sparsity=k,
            epsilon=eps,
            random_start=random_start
    )

    advAcc = DMP.validateD(advLoader, model, device)
    print("Adversarial Accuracy (L0+Linf PGD):", advAcc)


# =========================================================
# Attack 6: APGD-L1 on ResNet20-C model
# =========================================================
def APGD_L1_ResNet20():
    device, model, correctLoader = Load_ResNet20()

    # AutoAttack expects tensors (x, y)
    x_list, y_list = [], []
    for x, y in correctLoader:
        x_list.append(x)
        y_list.append(y)
    x_1k = torch.cat(x_list, dim=0)   # [1000,1,40,50]
    y_1k = torch.cat(y_list, dim=0)   # [1000]

    
    from l1_attack import AutoAttack   # works __init__.py exposes AutoAttack

    adv = AutoAttack(
        model,
        eps=2000,
        version="custom",
        attacks_to_run=["apgd-dlr"]
    )

    adv.apgd.n_iter = 500    
    adv.apgd.n_restarts = 1
    adv.apgd.use_rs = False
    adv.seed = 0

    print("\n--- Running APGD-L1 (DLR) ---")
    x_adv = adv.run_standard_evaluation(x_1k.to(device), y_1k.to(device), bs=32)

    adv_loader = DMP.TensorToDataLoader(x_adv.detach().cpu(), y_1k.cpu(), batchSize=32)
    adv_acc = DMP.validateD(adv_loader, model, device)
    print(f"[L1 APGD-DLR] Robust accuracy on 1k adv: {adv_acc*100:.2f}%")

