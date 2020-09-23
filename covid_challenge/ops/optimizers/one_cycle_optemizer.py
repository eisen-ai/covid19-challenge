import torch


class OneCycleOpt():
    r'''
    One cycle learning rate optimizer and sceduler class: the learning rate and momentum are adapted using the torch.optim.lr_scheduler.OneCycleLR()
    on a batchwise schedule. Implemented in a way that is compatible with the fixed input parameters of the eisen workflows.

    An instance of the optimizer (torch.optim.Adam) is required by the scheduler.

    Arguments: (OneCycleLR:)

      total_steps (int): The total number of steps in the cycle. Note that
      if a value is not provided here, then it must be inferred by providing
      a value for epochs and steps_per_epoch.
      Default: None

      anneal_strategy (str): {'cos', 'linear'}
      Specifies the annealing strategy: "cos" for cosine annealing, "linear" for
      linear annealing.
      Default: 'cos'

      pct_start (float): The percentage of the cycle (in number of steps) spent
      increasing the learning rate.
      Default: 0.3

      div_factor (float): Determines the initial learning rate via
      initial_lr = max_lr/div_factor
      Default: 25

      final_div_factor (float): Determines the minimum learning rate via
      min_lr = initial_lr/final_div_factor
      Default: 1e4

    Arguments: (Adam:)

      params (iterable): iterable of parameters to optimize or dicts defining
      parameter groups

      lr (float, optional): learning rate (default: 1e-3)

      betas (Tuple[float, float], optional): coefficients used for computing
      running averages of gradient and its square (default: (0.9, 0.999))

      eps (float, optional): term added to the denominator to improve
      numerical stability (default: 1e-8)

      weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

      amsgrad (boolean, optional): whether to use the AMSGrad variant of this
      algorithm from the paper `On the Convergence of Adam and Beyond`_
      (default: False)
    '''

    def __init__(self, total_steps, params, anneal_strategy='cos', lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, pct_start=0.3,div_factor=25.,final_div_factor=1e4):
        self.total_steps = total_steps
        self.anneal_strategy = anneal_strategy
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        self.optimizer = torch.optim.Adam(self.params, self.lr, self.betas, self.eps, self.weight_decay, self.amsgrad)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr,
                                                             total_steps=self.total_steps,
                                                             anneal_strategy=self.anneal_strategy,
                                                             pct_start=self.pct_start,
                                                             div_factor=self.div_factor,
                                                             final_div_factor=self.final_div_factor)

        self.param_groups = self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
        self.scheduler.step()
