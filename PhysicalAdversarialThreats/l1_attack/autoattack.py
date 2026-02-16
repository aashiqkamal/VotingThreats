# autoattack_l1.py
import math
import time
import numpy as np
import torch

from .other_utils import Logger
from . import checks
from .state import EvaluationState
from .autopgd_base import APGDAttack  # only dependency we need for attacks


class AutoAttackL1:
    """
    Minimal AutoAttack-style orchestrator that ONLY supports:
      - APGD on Cross-Entropy loss  ('apgd-ce')
      - APGD on DLR loss            ('apgd-dlr')
    Norm is fixed to 'L1'. No FAB / Square / targeted variants.

    Versions:
      - 'standard': runs apgd-ce then apgd-dlr, APGD n_restarts=5 (L1 default)
      - 'plus':     same attacks, but you can bump iterations/restarts if you want
      - 'rand':     same attacks with eot_iter=20 (randomized defenses)
      - 'custom':   respects attacks_to_run you pass in (subset of the two above)
    """

    def __init__(
        self,
        model,
        eps,
        seed=None,
        verbose=True,
        attacks_to_run=None,
        version="standard",
        device="cuda",
        log_path=None,
        n_iter=100,
        n_restarts=None,      # if None we'll pick defaults below
        eot_iter=1,
        rho=0.75
    ):
        self.model = model
        self.norm = "L1"
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.version = version
        self.device = device
        self.logger = Logger(log_path)

        # attacks we allow
        self._ALL = ["apgd-ce", "apgd-dlr"]
        self.attacks_to_run = list(attacks_to_run) if attacks_to_run else []

        # APGD (single instance; we switch the loss between 'ce' and 'dlr')
        self.apgd = APGDAttack(
            self.model,
            n_restarts=5 if n_restarts is None else n_restarts,
            n_iter=n_iter,
            verbose=False,
            eps=self.epsilon,
            norm=self.norm,
            eot_iter=eot_iter,
            rho=rho,
            seed=self.seed,
            device=self.device,
            logger=self.logger,
        )

        # Map version presets (and validate attacks_to_run)
        self._configure_version()

    # ------------------------ helpers ------------------------

    def get_logits(self, x):
        return self.model(x)

    def get_seed(self):
        return time.time() if self.seed is None else self.seed

    def _configure_version(self):
        if self.version in ["standard", "plus", "rand"] and self.attacks_to_run:
            raise ValueError(
                "attacks_to_run will be overridden unless you use version='custom'"
            )

        if self.version == "standard":
            # Classic ordering
            self.attacks_to_run = ["apgd-ce", "apgd-dlr"]
            # APGD defaults for L1
            self.apgd.n_restarts = 5
            self.apgd.eot_iter = 1

        elif self.version == "plus":
            self.attacks_to_run = ["apgd-ce", "apgd-dlr"]
            self.apgd.n_restarts = 5  # 
            self.apgd.eot_iter = 1

        elif self.version == "rand":
            self.attacks_to_run = ["apgd-ce", "apgd-dlr"]
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20  # stochastic defenses

        elif self.version == "custom":
            # 
            if not self.attacks_to_run:
                raise ValueError("With version='custom', provide attacks_to_run.")
            unsupported = [a for a in self.attacks_to_run if a not in self._ALL]
            if unsupported:
                raise ValueError(f"Unsupported attacks in attacks_to_run: {unsupported}")
        else:
            raise ValueError(f"Unknown version: {self.version}")

        if self.verbose:
            self.logger.log(
                f"Configured version '{self.version}' with attacks: {', '.join(self.attacks_to_run)}"
            )

    # ------------------------ public API ------------------------

    @torch.no_grad()
    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = math.ceil(x_orig.shape[0] / bs)
        acc = 0.0
        for counter in range(n_batches):
            x = x_orig[counter * bs : min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs : min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()

        if self.verbose:
            self.logger.log(f"clean accuracy: {acc / x_orig.shape[0]:.2%}")
        return acc.item() / x_orig.shape[0]

    def run_standard_evaluation(
        self,
        x_orig,
        y_orig,
        bs=250,
        return_labels=False,
        state_path=None,
    ):
        """
        Runs APGD-CE then APGD-DLR (or whatever is in attacks_to_run), only on
        currently robust points, like AutoAttack's choreography. Supports resuming via state.
        """
        # ----- state (optional resume) -----
        if state_path is not None and state_path.exists():
            state = EvaluationState.from_disk(state_path)
            if set(self.attacks_to_run) != state.attacks_to_run:
                raise ValueError(
                    "The state was created with a different set of attacks to run."
                )
            if self.verbose:
                self.logger.log(f"Restored state from {state_path}")
                self.logger.log(
                    "Since the state has been restored, only adversarials from the current run are returned."
                )
        else:
            state = EvaluationState(set(self.attacks_to_run), path=state_path)
            state.to_disk()
            if self.verbose and state_path is not None:
                self.logger.log(f"Created state in {state_path}")

        attacks_to_run = [a for a in self.attacks_to_run if a not in state.run_attacks]
        if self.verbose:
            self.logger.log(
                f"using {self.version} version including {', '.join(attacks_to_run)}."
            )
            if state.run_attacks:
                self.logger.log(f"{', '.join(state.run_attacks)} was/were already run.")

        # ----- pre-checks -----
        if self.version != "rand":
            checks.check_randomized(
                self.get_logits, x_orig[:bs].to(self.device), y_orig[:bs].to(self.device), bs=bs, logger=self.logger
            )
        n_cls = checks.check_range_output(self.get_logits, x_orig[:bs].to(self.device), logger=self.logger)
        checks.check_dynamic(self.model, x_orig[:bs].to(self.device), is_tf_model=False, logger=self.logger)
        # keep a minimal class-count sanity (no targeted logic here)
        if n_cls < 2:
            raise ValueError("Model appears to have < 2 classes from output range check.")

        # ----- initial clean evaluation -----
        with torch.no_grad():
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            if state.robust_flags is None:
                robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
                y_adv = torch.empty_like(y_orig)
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, x_orig.shape[0])
                    x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                    y = y_orig[start_idx:end_idx].clone().to(self.device)
                    output = self.get_logits(x).max(dim=1)[1]
                    y_adv[start_idx:end_idx] = output
                    robust_flags[start_idx:end_idx] = y.eq(output)

                state.robust_flags = robust_flags
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {"clean": robust_accuracy}
                state.clean_accuracy = robust_accuracy
                if self.verbose:
                    self.logger.log(f"initial accuracy: {robust_accuracy:.2%}")
            else:
                robust_flags = state.robust_flags.to(x_orig.device)
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {"clean": state.clean_accuracy}
                if self.verbose:
                    self.logger.log(f"initial clean accuracy: {state.clean_accuracy:.2%}")
                    self.logger.log(
                        f"robust accuracy at the time of restoring the state: {robust_accuracy:.2%}"
                    )

            x_adv = x_orig.clone().detach()
            startt = time.time()

            # ----- main loop over attacks -----
            for attack in attacks_to_run:
                num_robust = int(torch.sum(robust_flags).item())
                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))
                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)
                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)

                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)
                    if len(x.shape) == 3:
                        x = x.unsqueeze(0)

                    # run APGD with desired loss
                    if attack == "apgd-ce":
                        self.apgd.loss = "ce"
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y)
                    elif attack == "apgd-dlr":
                        self.apgd.loss = "dlr"
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y)
                    else:
                        raise ValueError(f"Attack not supported: {attack}")

                    output = self.get_logits(adv_curr).max(dim=1)[1]
                    false_batch = ~y.eq(output).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False
                    state.robust_flags = robust_flags

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                    # store adversarial labels (optional; mirrors original behavior)
                    y_adv[non_robust_lin_idcs] = output[false_batch].detach().to(x_adv.device)

                    if self.verbose:
                        num_non_robust_batch = int(torch.sum(false_batch).item())
                        self.logger.log(
                            f"{attack} - {batch_idx + 1}/{n_batches} - {num_non_robust_batch} out of {x.shape[0]} successfully perturbed"
                        )

                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict[attack] = robust_accuracy
                state.add_run_attack(attack)
                if self.verbose:
                    self.logger.log(
                        f"robust accuracy after {attack.upper()}: {robust_accuracy:.2%} (total time {time.time() - startt:.1f} s)"
                    )

            # ----- final checks and state save -----
            state.to_disk(force=True)
            if self.verbose:
                # L1 radius report
                res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).sum(dim=-1)
                self.logger.log(
                    f"max {self.norm} perturbation: {res.max():.5f}, "
                    f"nan in tensor: {(x_adv != x_adv).sum()}, max: {x_adv.max():.5f}, min: {x_adv.min():.5f}"
                )
                final_ra = torch.sum(robust_flags).item() / x_orig.shape[0]
                self.logger.log(f"robust accuracy: {final_ra:.2%}")

        return (x_adv, y_adv) if return_labels else x_adv

    # Convenience API mirroring upstream behavior
    def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250, return_labels=False):
        if self.verbose:
            self.logger.log(f"using {self.version} version including {', '.join(self.attacks_to_run)}")

        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False
        for attack in self.attacks_to_run:
            startt = time.time()
            x_adv, y_adv = self.run_standard_evaluation(x_orig, y_orig, bs=bs, return_labels=True)
            adv[attack] = (x_adv, y_adv) if return_labels else x_adv
            if verbose_indiv:
                acc_indiv = self.clean_accuracy(x_adv, y_orig, bs=bs)
                self.logger.log(
                    f"robust accuracy by {attack.upper()}\t {acc_indiv:.2%}\t (time attack: {time.time() - startt:.1f} s)"
                )
        return adv
