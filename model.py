import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torchinfo import summary
import torchaudio.transforms as T
from utils import ArcMarginProduct
import separableconv.nn as sep
import numpy as np
from mobilenet import MobileFaceNet

class AE_DCASEBaseline(nn.Module):
    def __init__(self) -> None:
        super(AE_DCASEBaseline, self).__init__()
        self.frames = 5
        self.n_mels = 128
        self.t_bins = 1 + (10 * 16000) // 512
        self.vector_array_size = self.t_bins - self.frames + 1
        self.transform_tf = T.MelSpectrogram(sample_rate=16000,
                                                n_fft=1024,
                                                win_length=1024,
                                                hop_length=512,
                                                center=True,
                                                pad_mode="reflect",
                                                power=2.0,
                                                norm="slaney",
                                                n_mels=self.n_mels,
                                                mel_scale="htk",
                                                ) 
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features = 640, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU()            
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(in_features = 128, out_features = 8),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features = 8, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 640)
        )

        # WEIGHTS INIT LIKE KERAS
        nn.init.xavier_uniform_(self.encoder[0].weight)
        nn.init.xavier_uniform_(self.encoder[3].weight)
        nn.init.xavier_uniform_(self.encoder[6].weight)
        nn.init.xavier_uniform_(self.encoder[9].weight)
        nn.init.xavier_uniform_(self.bottleneck[0].weight)
        nn.init.xavier_uniform_(self.decoder[0].weight)
        nn.init.xavier_uniform_(self.decoder[3].weight)
        nn.init.xavier_uniform_(self.decoder[6].weight)
        nn.init.xavier_uniform_(self.decoder[9].weight)
        nn.init.xavier_uniform_(self.decoder[-1].weight)

        # BIAS INIT LIKE KERAS
        nn.init.zeros_(self.encoder[0].bias)
        nn.init.zeros_(self.encoder[3].bias)
        nn.init.zeros_(self.encoder[6].bias)
        nn.init.zeros_(self.encoder[9].bias)
        nn.init.zeros_(self.bottleneck[0].bias)
        nn.init.zeros_(self.decoder[0].bias)
        nn.init.zeros_(self.decoder[3].bias)
        nn.init.zeros_(self.decoder[6].bias)
        nn.init.zeros_(self.decoder[9].bias)
        nn.init.zeros_(self.decoder[-1].bias)
        
    def forward(self, x):
        x = self.preprocessing(x)
        original = x
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return original, x
    
    def preprocessing(self, x):
        # compute mel spectrogram
        batch_size = x.size()[0]
        x = self.transform_tf(x)
        x = 10 * torch.log10(x + 1e-8)
        vector_dim = self.frames * self.n_mels
        
        feature_vector = torch.zeros((batch_size, self.vector_array_size, vector_dim)).to(x.device)
        for batch in range(batch_size):
            for i in range(self.frames):
                feature_vector[batch, :,  self.n_mels * i: self.n_mels * (i + 1)] = x[batch, :, i: i + self.vector_array_size].T
        
        return feature_vector
      
    
##### WAVEGRAM + MEL SPEC + ATTENTION MODULE + MOBILENETV2
class Wavegram_AttentionModule(nn.Module):
    def __init__(self, h):
        super(Wavegram_AttentionModule, self).__init__()
        self.h = h
        # sep-wavegram
        self.wavegram = sep.SeparableConv1d(in_channels = 1, out_channels = 128, kernel_size = 1024, stride = 512, padding = 512)
        # Mel filterbank
        self.transform_tf = T.MelSpectrogram(sample_rate=16000,
                                                n_fft=1024,
                                                win_length=1024,
                                                hop_length=512,
                                                center=True,
                                                pad_mode="reflect",
                                                power=2.0,
                                                norm="slaney",
                                                n_mels=128,
                                                mel_scale="htk",
                                                )
                                                
        # attention module
        self.heatmap = nn.Sequential(
                sep.SeparableConv2d(in_channels = 2, out_channels = 16,
                         kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                sep.SeparableConv2d(in_channels = 16, out_channels = 64, 
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(64),
                nn.ELU(),
                sep.SeparableConv2d(in_channels = 64, out_channels = 2, kernel_size = 1, padding = "same"),
                nn.Sigmoid()
            ) 
        # classifier
        self.classifier = MobileFaceNet(num_class = 41)
        self.arcface = ArcMarginProduct(in_features = self.h, out_features = 41, s = 40, m = 0.7)
    
    def forward(self, x, metadata):
        # compute mel spectrogram
        x_spec = self.transform_tf(x)
        x_spec = 10*torch.log10(x_spec + 1e-8)
        # compute wavegram
        x = x.unsqueeze(1)
        x = self.wavegram(x)
        x = torch.stack((x_spec, x), dim = 1)
        reppr = x
        heatmap = self.heatmap(x)
        x = x * heatmap
        out, features = self.classifier(x)
        x = self.arcface(features, metadata)
        return x, out, reppr, features, heatmap

class Wavegram_AttentionMap(LightningModule):

    def __init__(self, h, lr):
        super().__init__()
        self.h = h
        self.model = Wavegram_AttentionModule(self.h)
        self.lr = lr

        self.accuracy_training = Accuracy(task="multiclass", num_classes=41)
        self.accuracy_val = Accuracy(task="multiclass", num_classes=41)
        self.accuracy_test = Accuracy(task="multiclass", num_classes=41)
        self.criterion = nn.CrossEntropyLoss()
        # to save threshold and errors at init
        self.errors_list = []
        self.clean_errors = []
        self.anomaly_errors = []
        self.labels = []
        self.classes = []
    
    def mixup_data(self, x, y, alpha=0.2):
        y = torch.nn.functional.one_hot(y, num_classes = 41)
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a.float(), y_b.float(), lam

    def forward(self, x, labels):
        return self.model(x, labels)
        
    def mixup_criterion_arcmix(self, pred, y_a, y_b, lam):
        loss1 = lam * self.criterion(pred, y_a)
        loss2 = (1 - lam) * self.criterion(pred, y_b)
        return loss1+loss2
    
    def training_step(self, batch, batch_idx):
        x, metadata, _, _ = batch
        # for training step
        mixed_x, y_a, y_b, lam = self.mixup_data(x, metadata)
        predicted, _, _, _, _ = self.forward(mixed_x, metadata)
        loss = self.mixup_criterion_arcmix(predicted, y_a, y_b, lam)
        self.log("train/loss_class", loss, on_epoch = True, on_step = True, prog_bar = True)
        self.accuracy_training(predicted, metadata)
        self.log("train/acc", self.accuracy_training, on_epoch = True, on_step = False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, metadata, _, _ = batch
        predicted, _, _, _, _ = self.forward(x, metadata)
        loss = torch.nn.functional.cross_entropy(predicted, metadata, reduction = "mean")
        self.log("val/loss_class", loss, on_epoch = True, on_step = False, prog_bar = True)
        self.accuracy_val(predicted, metadata)
        self.log("val/acc", self.accuracy_val, on_epoch = True, on_step = False)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, metadata, label, _ = batch
        predicted, _, _, _, _ = self.forward(x, metadata)
        loss = torch.nn.functional.cross_entropy(predicted, metadata, reduction = "mean")
        self.log("test/loss_class", loss, on_epoch = True, on_step = False, prog_bar = True)
        self.accuracy_test(predicted, metadata)
        self.log("test/acc", self.accuracy_test, on_epoch = True, on_step = False)
        class_loss_batchwise = nn.functional.cross_entropy(predicted, metadata, reduction = "none") 
        errors = class_loss_batchwise
        self.errors_list.append(errors)
        self.clean_errors.append(errors[label == 0])
        self.anomaly_errors.append(errors[label == 1])
        self.labels.append(label)
        self.classes.append(metadata)
        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr = self.lr)
        # return opt
        return {
           "optimizer": opt,
           "lr_scheduler": {
               "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=300, eta_min=0.1*float(self.lr))
                           },
              }   
    
    
# TEST FUNCTION
if __name__ == "__main__":
    example_input = torch.rand(16, 160000) # dummy audio
    model = Wavegram_AttentionModule()
    metadata = torch.nn.functional.one_hot(torch.randint(low = 0, high = 41, size =(16,)), num_classes=41)
    output = model(example_input, metadata)
    print(output)
    summary(model, input_data = example_input)
    
