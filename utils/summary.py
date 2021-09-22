from tensorboardX import SummaryWriter
import os

__all__ = ["Summary"]


class Summary(object):
    def __init__(self, logdir):
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        self.writer = None
        self.writer = SummaryWriter(logdir=logdir)

    def __del__(self):
        if self.writer:
            self.writer.close()

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_graph(self, model, input_to_model=None, verbose=False):
        self.writer.add_graph(model, input_to_model, verbose)
