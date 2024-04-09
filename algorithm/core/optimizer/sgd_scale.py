import torch

class SGDInt(torch.optim.SGD):
    @staticmethod
    def post_step(model):
        from quantize.quantized_ops_diff import QuantizedConv2dDiff
        for m in model.modules():
            if isinstance(m, QuantizedConv2dDiff):
                if m.bias.grad is not None:
                    m.bias.data = m.bias.data.round()
                if m.weight.grad is not None:
                    m.weight.data = m.weight.data.round()

class SGDScale(torch.optim.SGD):
    @staticmethod
    def pre_step(model):
        # add a pre_step method to scale the gradient, since sometimes we need information from the model,
        # but not only parameters.
        from quantize.quantized_ops_diff import QuantizedConv2dDiff
        for m in model.modules():
            if isinstance(m, QuantizedConv2dDiff):
                if m.bias.grad is not None:
                    m.bias.grad.data = m.bias.grad.data / (m.effective_scale.data * m.y_scale) ** 2
                if m.weight.grad is not None:
                    w_scale = m.effective_scale.data * m.y_scale / m.x_scale
                    m.weight.grad.data = m.weight.grad.data / w_scale.view(-1, 1, 1, 1) ** 2

class SGDScaleInt(torch.optim.SGD):
    @staticmethod
    def pre_step(model):
        # add a pre_step method to scale the gradient, since sometimes we need information from the model,
        # but not only parameters.
        from quantize.quantized_ops_diff import QuantizedConv2dDiff
        for m in model.modules():
            if isinstance(m, QuantizedConv2dDiff):
                if m.bias.grad is not None:
                    m.bias.grad.data = m.bias.grad.data / (m.effective_scale.data * m.y_scale) ** 2
                if m.weight.grad is not None:
                    w_scale = m.effective_scale.data * m.y_scale / m.x_scale
                    m.weight.grad.data = m.weight.grad.data / w_scale.view(-1, 1, 1, 1) ** 2
    
    @staticmethod
    def post_step(model):
        from quantize.quantized_ops_diff import QuantizedConv2dDiff
        for m in model.modules():
            if isinstance(m, QuantizedConv2dDiff):
                if m.bias.grad is not None:
                    m.bias.data = m.bias.data.round()
                if m.weight.grad is not None:
                    m.weight.data = m.weight.data.round()

