from mmcv.runner.hooks import Hook
from mmcv.runner.utils import master_only


class Every_n_iter_CheckpointHook(Hook):

    def __init__(self,
                 interval=10000,
                 save_optimizer=True,
                 out_dir=None,
                 **kwargs):
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.args = kwargs

    @master_only
    def after_train_iter(self, runner):
        # for name, param in runner.model.module.named_parameters():
        #     if param.grad is None:
        #         print(name)
        # print('________________________________________')
        if self.every_n_iters(runner,self.interval):
            if not self.out_dir:
                self.out_dir = runner.work_dir
            runner.save_checkpoint(
                self.out_dir, filename_tmpl='epoch_'+str(runner.iter)+'_{}.pth', save_optimizer=self.save_optimizer, **self.args)