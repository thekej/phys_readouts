import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchmetrics

from pytorch_lightning import LightningModule, Trainer

from models.readout_models import lr, self_attention_cnn, cnn, attention_mlp, mlp


class BinaryClassificationWrapper(LightningModule):
    def __init__(self, model, weight_decay, lr, use_lr_schedule=False):
        super().__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.lr = lr
        self.use_lr_schedule = use_lr_schedule

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy(y_hat.squeeze(1), y)
        accuracy = torch.mean(((y_hat.squeeze(1) > 0.5).float() == y).float())
        self.log('train_acc', accuracy, on_step=True, prog_bar=True)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy(y_hat.squeeze(1), y)#.unsqueeze(1))
        accuracy = torch.mean(((y_hat.squeeze(1) > 0.5).float() == y).float())
        self.log('val_acc', accuracy, on_step=True, prog_bar=True)
        self.log("val_loss", loss)
        return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.unsqueeze(1))
        accuracy = torch.mean(((y_hat.squeeze(1) > 0.5).float() == y).float())
        self.log('test_loss', loss, on_step=True)
        self.log('test_acc', accuracy, on_step=True)
        return {'test_loss': loss}
    
    def configure_optimizers(self):
        #optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay) 
        optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr, max_iter=50)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, 
                                                               patience=10, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        #return optimizer
                                      

def get_model(model_name, input_size, weight_decay=1e-1, learning_rate=1e-3, scheduler=False):
    if model_name == 'LR':
        print(input_size)
        model = lr.BinaryLogisticRegression(input_size)
    if model_name == 'CNN':
        model = cnn.CNN(input_size)
    if model_name == 'MLP':
        model = mlp.MLP(input_size)
    if model_name == 'ATTENTION_MLP':
        model = attention_mlp.BinaryAttentionModel(input_size)
    if model_name == 'ATTENTION_CNN':
        model = self_attention_cnn.CNNWithAttention(input_size)
        
    return BinaryClassificationWrapper(model, weight_decay, learning_rate, scheduler)