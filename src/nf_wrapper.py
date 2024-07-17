
def assign_params_to_flow(flow, flat_params):
    parameters = flow.parameters()
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        param_flattened = param.flatten()
        for i in range(num_param):
            param_flattened[i] = flat_params[pointer + i]
        pointer += num_param


def sample_pullback(flow, flat_parameters, num_samples):
    flat_parameters.requires_grad = True
    assign_params_to_flow(flow, flat_parameters)
    samples = flow.sample(num_samples)[0].t()

    def pullback_f(y):
        samples.backward(y, retain_graph=True)
        grad = flat_parameters.grad
        return grad

    return samples.detach().clone().numpy(), pullback_f


def logpdf_pullback(flow, flat_parameters, x):
    flat_parameters.requires_grad = True
    assign_params_to_flow(flow, flat_parameters)
    log_prob = flow.log_prob(x).t()

    def pullback_f(y):
        log_prob.backward(y, retain_graph=True)
        grad = flat_parameters.grad
        return grad

    return log_prob.detach().clone().numpy(), pullback_f
