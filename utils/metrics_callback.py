from pytorch_lightning.callbacks import Callback

class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics['train_loss'].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_losses.append(trainer.callback_metrics['val_loss'].item())
        self.val_accs.append(trainer.callback_metrics['val_acc'].item())


    def on_train_start(self, trainer, pl_module):
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
