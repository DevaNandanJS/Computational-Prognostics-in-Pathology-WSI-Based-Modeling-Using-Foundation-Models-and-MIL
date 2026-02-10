import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd

#---->

from MyLoss import create_loss
from utils.utils import cross_entropy_torch

#---->
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

#---->
import pytorch_lightning as pl


class  ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.n_classes = model['n_classes']
        self.log_path = kargs['log']

        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        #---->Metrics
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(num_classes = self.n_classes, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = self.n_classes,
                                                                           average='micro'),
                                                     torchmetrics.CohenKappa(num_classes = self.n_classes),
                                                     torchmetrics.F1(num_classes = self.n_classes,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = self.n_classes),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = self.n_classes),
                                                     torchmetrics.Specificity(average = 'macro',
                                                                            num_classes = self.n_classes)])
        else : 
            self.AUROC = torchmetrics.AUROC(task='binary', num_classes=2, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='binary', num_classes = 2,
                                                                           average = 'micro'),
                                                     torchmetrics.CohenKappa(task='binary', num_classes = 2),
                                                     torchmetrics.F1Score(task='binary', num_classes = 2,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(task='binary', average = 'macro',
                                                                         num_classes = 2),
                                                     torchmetrics.Precision(task='binary', average = 'macro',
                                                                            num_classes = 2)])
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        #--->random
        self.shuffle = kargs['data']['data_shuffle']
        self.count = 0


    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        #---->inference
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        #---->loss
        loss = self.loss(logits, label)

        #---->acc log
        Y_hat = int(Y_hat)
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

        return {'loss': loss} 

    def training_epoch_end(self, training_step_outputs):
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def validation_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']


        #---->acc log
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}


    def validation_epoch_end(self, val_step_outputs):
        # --- Start: Keep all existing metric calculation logic ---
        logits = torch.cat([x['logits'] for x in val_step_outputs], dim=0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim=0)
        max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
        target = torch.stack([x['label'] for x in val_step_outputs], dim=0)

        val_loss = self.loss(logits, target)
        val_auc = self.AUROC(probs[:, 1], target.squeeze())
        
        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True, logger=True)
        self.log('auc', val_auc, prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(max_probs.squeeze(), target.squeeze()),
                      on_epoch=True, logger=True)
        # --- End: Keep all existing metric calculation logic ---

        # --- Start: New Detailed and Clean Logging ---
        # Fetch metrics (use locally calculated values for robustness)
        current_val_loss = val_loss
        current_auc = val_auc
        
        # Safely get the best score and handle the case where it's not set yet
        best_val_loss = self.trainer.checkpoint_callback.best_model_score
        if best_val_loss is None:
            best_val_loss = float('inf')
            
        # Get current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        # Prepare log message
        print(f"\n\n{'='*60}")
        print(f"  Epoch {self.trainer.current_epoch} Validation Summary")
        print(f"{'-'*60}")
        
        if current_val_loss is not None:
            print(f"    Validation Loss: {current_val_loss:.5f}")
        if current_auc is not None:
            print(f"    Validation AUC:  {current_auc:.5f}")
            
        print(f"\n    Best Validation Loss: {best_val_loss:.5f}")

        # Check if the model was saved in this epoch
        if current_val_loss is not None and current_val_loss < best_val_loss:
            print("    --> Validation Loss Improved! Model checkpoint SAVED.")
        else:
            print("    --> Model checkpoint was NOT saved this epoch.")
            
        print(f"\n    Current Learning Rate: {current_lr:.1e}")
        print(f"{'='*60}\n")
        # --- End: New Detailed and Clean Logging ---

        # --- Start: Keep the random seed logic ---
        if self.shuffle == True:
            self.count = self.count + 1
            random.seed(self.count * 50)
        # --- End: Keep the random seed logic ---
    


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.optimizer['lr'], weight_decay=self.hparams.optimizer['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def test_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        #---->acc log
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    def test_epoch_end(self, output_results):
        logits = torch.cat([x['logits'] for x in output_results], dim=0) # Need logits for loss
        probs = torch.cat([x['Y_prob'] for x in output_results], dim=0)
        max_probs = torch.stack([x['Y_hat'] for x in output_results])
        target = torch.stack([x['label'] for x in output_results], dim=0)
        
        # Calculate test loss, as it's not currently calculated in test_step
        test_loss = self.loss(logits, target)

        auc = self.AUROC(probs[:, 1], target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())
        metrics['auc'] = auc # Add AUC to the metrics dictionary

        # --- Start: New Detailed and Clean Logging ---
        print(f"\n\n{'='*60}")
        print(f"  Final Test Report")
        print(f"{'-'*60}")
        
        print(f"    Test Loss: {test_loss:.5f}")
        print(f"    Test AUC:  {auc:.5f}")
        
        # Print other relevant test metrics
        for key, value in metrics.items():
            if key != 'auc': # Already printed AUC separately
                print(f"    {key}: {value:.5f}")

        print(f"\n    Class-wise Accuracy:")
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = "N/A"
            else:
                acc = f"{float(correct) / count:.3f}"
            print(f"      class {c}: acc {acc}, correct {correct}/{count}")
        
        print(f"{'='*60}\n")
        # --- End: New Detailed and Clean Logging ---

        # Reset self.data for class-wise accuracy tracking
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        # Convert metrics values to numpy for CSV saving
        for keys, values in metrics.items():
            metrics[keys] = values.cpu().numpy()

        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path / 'result.csv')


    def load_model(self):
        name = self.hparams.model['name']
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.hparams.model[arg]
        args1.update(other_args)
        return Model(**args1)