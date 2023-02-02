import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
        
        loss = F.binary_cross_entropy(y_hat, y.unsqueeze(1))
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy(y_hat, y.unsqueeze(1))
        accuracy = torch.mean((torch.round(torch.sigmoid(y_hat)) == y.unsqueeze(1)).float())
        self.log('val_accuracy', accuracy, on_step=True)
        self.log("val_loss", loss)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.unsqueeze(1))
        accuracy = torch.mean((torch.round(torch.sigmoid(y_hat)) == y.unsqueeze(1)).float())
        self.log('test_loss', loss, on_step=True)
        self.log('test_accuracy', accuracy, on_step=True)
        return {'test_loss': loss}
 
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)  
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return [optimizer], [scheduler]
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay) 
        if self.use_lr_schedule:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                                   patience=10, verbose=True, threshold=0.0001, 
                                                                   threshold_mode='rel', cooldown=0, 
                                                                   min_lr=0, eps=1e-08)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return optimizer
                                      

def get_model(model_name, input_size, weight_decay=1e-1, learning_rate=1e-3, pretrained=None, scheduler=False):
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
        
    # Load the saved file
    if not pretrained is None:
        return BinaryClassificationWrapper.load_from_checkpoint(pretrained, model=model)
        
    return BinaryClassificationWrapper(model, weight_decay, learning_rate, scheduler)




    