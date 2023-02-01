import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pytorch_lightning import LightningModule, Trainer

from models.readout_models import lr, self_attention_cnn, cnn, attention_mlp, mlp


class BinaryClassificationWrapper(LightningModule):
    def __init__(self, model, weight_decay, lr):
        super().__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy = torch.mean((torch.round(torch.sigmoid(y_hat)) == y).float())
        self.log('val_accuracy', accuracy, on_step=True)
        self.log("val_loss", loss)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy = torch.mean((torch.round(torch.sigmoid(y_hat)) == y).float())
        self.log('test_loss', loss, on_step=True)
        self.log('test_accuracy', accuracy, on_step=True)
        return {'test_loss': loss}


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)   
                                      

def get_model(model_name, input_size, weight_decay=1e-2, lr=1e-3, pretrained=None, train=True):
    if model_name == 'LR':
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
        
    return BinaryClassificationWrapper(model, weight_decay, lr)




    