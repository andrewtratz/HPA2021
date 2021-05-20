import torch.optim as optim
from layers.scheduler_base import SchedulerBase


class AdamFineTune(SchedulerBase):
    def __init__(self, params_list=None):
        super(AdamFineTune, self).__init__()
        self._lr = 15e-5
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self, net, epoch, epochs, **kwargs):
        lr = 15e-5
        if epoch > 25:
            lr = 7.5e-5
        if epoch > 30:
            lr = 3e-5
        if epoch > 35:
            lr = 1e-5
        if epoch > 40:
            lr = 1e-5
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.0005
        return self._cur_optimizer, self._lr

class Adam20(SchedulerBase):
    def __init__(self, params_list=None):
        super(Adam20, self).__init__()
        self._lr = 15e-5
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self, net, epoch, epochs, **kwargs):
        lr = 15e-5
        if epoch > 8:
            lr = 5e-5
        if epoch > 12:
            lr = 1e-5
        if epoch > 16:
            lr = 5e-6
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.0005
        return self._cur_optimizer, self._lr

class AdamCustom(SchedulerBase):
    def __init__(self, params_list=None):
        super(AdamCustom, self).__init__()
        self._lr = 15e-5
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self, net, epoch, epochs, **kwargs):
        if epoch >= 5:
            lr = 10e-5
        if epoch >= 6:
            lr = 5e-5
        if epoch >= 7:
            lr = 5e-5
        if epoch >= 8:
            lr = 1e-5
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.0005
        return self._cur_optimizer, self._lr

class Adam45(SchedulerBase):
    def __init__(self, params_list=None):
        super(Adam45, self).__init__()
        self._lr = 3e-4
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self, net, epoch, epochs, **kwargs):
        lr = 3e-4
        if epoch > 5:
            lr = 15e-5
        if epoch > 10:
            lr = 5e-5
        if epoch > 15:
            lr = 1e-5
        if epoch > 20:
            lr = 5e-6
        if epoch > 25:
            lr = 1e-6
        if epoch > 30:
            lr = 5e-7
        if epoch > 35:
            lr = 1e-7
        if epoch > 40:
            lr = 5e-8
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.0005
        return self._cur_optimizer, self._lr

class AdamAndrew35(SchedulerBase):
    def __init__(self, params_list=None):
        super(AdamAndrew35, self).__init__()
        self._lr = 3e-4
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self, net, epoch, epochs, **kwargs):
        lr = 3e-4
        if epoch > 2:
            lr = 15e-5
        if epoch > 10:
            lr = 1e-4
        if epoch > 15:
            lr = 5e-6
        if epoch > 20:
            lr = 1e-6
        if epoch > 25:
            lr = 5e-7
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.0005
        return self._cur_optimizer, self._lr

class AdamShortRun(SchedulerBase):
    def __init__(self, params_list=None):
        super(AdamShortRun, self).__init__()
        self._lr = 3e-4
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self, net, epoch, epochs, **kwargs):
        lr = 3e-4
        if epoch > 1:
            lr = 15e-5
        if epoch > 3:
            lr = 1e-4
        if epoch > 4:
            lr = 5e-5
        if epoch > 5:
            lr = 1e-5
        if epoch > 6:
            lr = 5e-6
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.0005
        return self._cur_optimizer, self._lr


class Adam55(SchedulerBase):
    def __init__(self, params_list=None):
        super(Adam55, self).__init__()
        self._lr = 3e-4
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self,net, epoch, epochs, **kwargs):
        lr = 30e-5
        if epoch > 25:
            lr = 15e-5
        if epoch > 35:
            lr = 7.5e-5
        if epoch > 45:
            lr = 3e-5
        if epoch > 50:
            lr = 1e-5
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.0005
        return self._cur_optimizer, self._lr

class FaceAdam(SchedulerBase):
    def __init__(self,params_list=None):
        super(FaceAdam, self).__init__()
        self._lr = 2e-4
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self, net, epoch, epochs, **kwargs):
        lr = 1e-4
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)  # , weight_decay=0.0005
        return self._cur_optimizer, self._lr
