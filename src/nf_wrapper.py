import copy
import torch


def make_functional_with_buffers(mod, disable_autograd_tracking=False):
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    buffers_dict = dict(mod.named_buffers())
    buffers_names = buffers_dict.keys()
    buffers_values = tuple(buffers_dict.values())

    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to("meta")

    def fmodel(new_params_values, new_buffers_values, *args, **kwargs):
        new_params_dict = {
            name: value for name, value in zip(params_names, new_params_values)
        }
        new_buffers_dict = {
            name: value for name, value in zip(buffers_names, new_buffers_values)
        }
        return torch.func.functional_call(
            stateless_mod, (new_params_dict, new_buffers_dict), args, kwargs
        )

    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values, buffers_values


def flatten_params(parameters):
    """
    flattens all parameters into a single column vector. Returns the dictionary to recover them
    :param: parameters: a generator or list of all the parameters
    :return: a dictionary: {"params": [#params, 1],
    "indices": [(start index, end index) for each param] **Note end index in uninclusive**

    """
    l = [torch.flatten(p) for p in parameters]
    indices = []
    s = 0
    for p in l:
        size = p.shape[0]
        indices.append((s, s + size))
        s += size
    flat = torch.cat(l)
    return flat, indices


def recover_flattened(flat_params, indices, parameters):
    """
    Gives a list of recovered parameters from their flattened form
    :param flat_params: [#params, 1]
    :param indices: a list detaling the start and end index of each param [(start, end) for param]
    :param model: the model that gives the params with correct shapes
    :return: the params, reshaped to the ones in the model, with the same order as those in the model
    """
    l = [flat_params[s:e] for (s, e) in indices]
    for i, p in enumerate(parameters):
        l[i] = l[i].view(*p.shape)
    return l


class SamplerWrapper(torch.nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, n):
        return self.flow.sample(n)[0].t()


class LogProbWrapper(torch.nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, x):
        return self.flow.log_prob(x.t()).t()


def deconstruct_flow(flow):
    sampler = SamplerWrapper(flow)
    log_prob = LogProbWrapper(flow)
    func_sampler, params, buffers = make_functional_with_buffers(sampler)
    func_log_prob, _, _ = make_functional_with_buffers(log_prob)
    return func_sampler, func_log_prob, params, buffers


def make_vjp_sampler(func_sampler, params, params_flat, buffers, indices, n_samples):
    def faux(params_flat):
        new_params = recover_flattened(params_flat, indices, params)
        return func_sampler(new_params, buffers, n_samples)

    return torch.func.vjp(faux, params_flat)


def make_vjp_logpdf(func_logpdf, params, params_flat, buffers, indices, samples):
    def faux(params_flat, x):
        new_params = recover_flattened(params_flat, indices, params)
        return func_logpdf(new_params, buffers, x)

    return torch.func.vjp(faux, params_flat, samples)
