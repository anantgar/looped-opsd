"""Full-batch CE gradient balancing; native forward/readout remains unchanged."""
import math
import torch


def coefficient(deep_norm, student_norm, lam, rho=.25, cap=1., epsilon=1e-12):
    if not all(math.isfinite(x) and x >= 0 for x in (deep_norm, student_norm, lam, rho, cap)):
        raise ValueError('Invalid gradient balance value')
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError('Invalid gradient norm epsilon')
    raw = lam*rho*deep_norm/max(student_norm, epsilon) if deep_norm and student_norm else 0.
    return min(cap, raw), raw


class GradientBalance:
    """Accumulate deep gradients on CPU; student gradients use parameter.grad.

    Both components carry the same GradScaler scale. Compute one detached scalar
    for the effective batch, combine while scaled, then let the caller unscale,
    clip and perform exactly one optimizer update.
    """
    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.deep = [None]*len(self.parameters)

    def backward(self, deep_loss, student_loss, scaler):
        gradients = torch.autograd.grad(scaler.scale(deep_loss), self.parameters,
                                       retain_graph=True, allow_unused=True)
        for i, gradient in enumerate(gradients):
            if gradient is None:
                continue
            value = gradient.detach().to('cpu')
            if self.deep[i] is None:
                self.deep[i] = value
            else:
                self.deep[i].add_(value)
        del gradients
        gradient = value = None
        scaler.scale(student_loss).backward()

    def combine(self, scale, lam, rho=.25, cap=1., epsilon=1e-12):
        a2 = b2 = dot = 0.
        # Bounded temporaries: never materialize a double-precision model vector.
        for parameter, deep in zip(self.parameters, self.deep):
            student = parameter.grad.detach().to('cpu') if parameter.grad is not None else None
            size = parameter.numel()
            for start in range(0, size, 1 << 20):
                a = deep.reshape(-1)[start:start+(1 << 20)].double() if deep is not None else None
                b = student.reshape(-1)[start:start+(1 << 20)].double() if student is not None else None
                if a is not None: a2 += float(torch.dot(a, a))
                if b is not None: b2 += float(torch.dot(b, b))
                if a is not None and b is not None: dot += float(torch.dot(a, b))
            del student
        deep_norm, student_norm = math.sqrt(a2)/scale, math.sqrt(b2)/scale
        alpha, raw = coefficient(deep_norm, student_norm, lam, rho, cap, epsilon)
        if not math.isfinite(dot): raise ValueError('Nonfinite gradient dot product')
        combined_norm = math.sqrt(max(0., a2+2*alpha*dot+alpha*alpha*b2))/scale
        stats = {'alpha': alpha, 'alpha_uncapped': raw, 'alpha_cap_hit': raw > cap,
                 'gradient_deep_norm': deep_norm, 'gradient_student_norm': student_norm,
                 'gradient_cosine': dot/math.sqrt(a2*b2) if a2 and b2 else 0.,
                 'gradient_cosine_defined': bool(a2 and b2),
                 'gradient_aux_deep_ratio': alpha*student_norm/deep_norm if deep_norm else 0.,
                 'gradient_zero_deep': deep_norm == 0., 'gradient_zero_student': student_norm == 0.,
                 'gradient_epsilon_active': 0 < student_norm < epsilon,
                 'gradient_combined_norm_estimate': combined_norm}
        with torch.no_grad():
            for i, parameter in enumerate(self.parameters):
                deep = self.deep[i]
                if parameter.grad is not None:
                    parameter.grad.mul_(alpha)
                    if deep is not None: parameter.grad.add_(deep.to(parameter.device))
                elif deep is not None:
                    parameter.grad = deep.to(parameter.device)
                self.deep[i] = None
        return stats
