from mmcv.runner.hooks import Hook


class Use_Poly_Loss_Hook(Hook):

    def __init__(self,
                 use_poly_loss_epoch=5,
                 **kwargs):
        self.use_poly_loss_epoch = use_poly_loss_epoch

    def before_epoch(self, runner):
        if runner._epoch >= self.use_poly_loss_epoch:
            runner.model.module.mask_head.use_poly_loss= True
            print('use poly loss:{}'.format(runner.model.module.mask_head.use_poly_loss))
        else:
            runner.model.module.mask_head.use_poly_loss = False
            print('use poly loss:{}'.format(runner.model.module.mask_head.use_poly_loss))