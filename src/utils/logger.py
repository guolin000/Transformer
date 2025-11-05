# src/utils/logger.py
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
    @property
    def avg(self):
        return self.sum / self.cnt if self.cnt else 0
