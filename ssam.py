import torch
import numpy as np
import projgrad
import cupy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ssam_obj_func(beta, nabla_f, nabla_l, lam):
    beta, nabla_f, nabla_l = cupy.ravel(beta), cupy.ravel(nabla_f), cupy.ravel(nabla_l)
    f = - (1 - lam)*cupy.inner(beta, nabla_l) - lam*(cupy.inner(beta,nabla_f))**2
    
    grad = -(1 - lam)*nabla_l - lam*2 * nabla_f * cupy.inner(nabla_f, beta) 
    return f, grad
    

class SSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, lam=0.5, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.rho = rho

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.nabla_f = {}
        self.lam = lam
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False, n_iter=25):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                nabla_f = self.nabla_f[p]
                nabla_l = p.grad
                beta_init = p.grad
                if torch.norm(beta_init, 2) != 0:
                    res = projgrad.minimize(ssam_obj_func, x0=cupy.ravel(cupy.array(beta_init.cpu().numpy())), 
                                            rho=self.rho, args=(cupy.array(nabla_f.cpu().numpy()), cupy.array(nabla_l.cpu().numpy()), self.lam), 
                                            maxiters=n_iter, algo=None, disp=False)
                    e_w = res.x
                else:
                    e_w = torch.zeros(p.shape)
                p.add_(torch.Tensor(np.reshape(e_w, p.shape)).to(device))  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def prep(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.nabla_f[p] = p.grad
        if zero_grad: self.zero_grad()


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