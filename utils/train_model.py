import torch
import pytorch_lightning as pl
from utils.model_data import MushroomDataModule, ResNetModel
from utils.eval import evaluate_model, plot_metrics
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils.metrics_callback import MetricsCallback
import os
from pathlib import Path

def train_model(model_class, batch_size, learning_rate, accumulation_steps=1, early_stop_patience=5):
    data_module = MushroomDataModule(data_dir="data/Mushrooms", batch_size=batch_size)
    data_module.setup() 

    model = model_class(num_classes=9, learning_rate=learning_rate, class_weights=data_module.class_weights)

    base_checkpoint_path = "models/" + model_class.__name__ + "_checkpoints"
    checkpoint_dir = os.path.join(base_checkpoint_path, f"batch_size={batch_size}_lr={learning_rate}_accum_steps={accumulation_steps}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min"
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=early_stop_patience)
    metrics_callback = MetricsCallback()

    trainer = Trainer(
        max_epochs=999,
        callbacks=[checkpoint_callback, early_stopping_callback, metrics_callback],
        logger=pl.loggers.TensorBoardLogger("lightning_logs", name=f"{model_class.__name__}-bs{batch_size}-lr{learning_rate}"),
        enable_progress_bar=True,
        accumulate_grad_batches=accumulation_steps
    )

    trainer.fit(model, data_module)

    plot_metrics(metrics_callback)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from {best_model_path} for evaluation")
        best_model = model_class.load_from_checkpoint(best_model_path)
        evaluate_model(best_model, data_module.val_dataloader(), "cuda" if torch.cuda.is_available() else "cpu")
    else:
        print("No best model checkpoint found.")
