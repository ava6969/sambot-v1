from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, dir: str):
        self.time_step = 0
        self.writer = SummaryWriter(dir, flush_secs=30, max_queue=5)

    def step(self):
        self.time_step += 1

    def write(self, name, data):
        self.writer.add_scalar(name, data, self.time_step)

    def close(self):
        self.writer.close()