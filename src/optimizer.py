import mindspore.nn as nn
import mindspore.ops as ops

class MyOptimizer(nn.Adam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_construct = super().construct
        self.histogram_summary = ops.HistogramSummary()
        self.gradient_names = [param.name + ".gradient" for param in self.parameters]

    def construct(self, grads):
        # Record gradient.
        self.histogram_summary(self.gradient_names[0], grads[0])
        return self._original_construct(grads)