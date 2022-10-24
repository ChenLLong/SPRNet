from mmcv.runner.hooks import Hook


class Exchange_Anchor_t_Hook(Hook):

    def __init__(self,
                 exchange_t_epoch=5,
                 by_epoch=True,
                 **kwargs):
        self.exchange_t_epoch = exchange_t_epoch

    def before_epoch(self, runner):
        if runner._epoch >= self.exchange_t_epoch:
            runner.model.module.mask_head.use_anchor_t = False
            print('use anchor t:{}'.format(runner.model.module.mask_head.use_anchor_t))
        else:
            runner.model.module.mask_head.use_anchor_t = True
            print('use anchor t:{}'.format(runner.model.module.mask_head.use_anchor_t))