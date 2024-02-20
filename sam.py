# Source: https://github.com/davda54/sam/blob/main/sam.py
import torch
import numpy.linalg as la
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.1, adaptive=False, second_order=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        # self.rho = rho
        self.second_order = second_order

    @torch.no_grad()
    def first_step(self, eigenvecs=None, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        # else:
        #     # model_param_count = 0
        #     # for group in self.param_groups:
        #     #     for p in group["params"]:
        #     #         if p.grad is not None:
        #     #             #if p not in model_param_dict.keys():
        #     #             #     model_param_dict[p] = {}
        #     #             # model_param_dict[p] = p.shape
        #     #             self.state[p]["old_p"] = p.data.clone()
        #     #             curr_param_size = int((torch.flatten(p)).size(dim=0))
        #     #             curr_eig_vec = torch.Tensor(eigenvecs[0][model_param_count:model_param_count+curr_param_size]) # get eigvec component corresp. to this parameter
        #     #             curr_eig_vec_norm = la.norm(curr_eig_vec, 2)
        #     #             e_w = self.rho * curr_eig_vec/curr_eig_vec_norm
        #     #             p.add_(torch.reshape(e_w.to(device), p.shape))
        #     #             model_param_count += curr_param_size
        #     proj grad 

        if zero_grad: self.zero_grad()

        return e_w
    
    def first_grad_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                new_p = p.clone()
                new_p.add_(e_w)  # climb to the local maximum "w + e(w)"
                p = new_p
        if zero_grad: self.zero_grad()
        return

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups