import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from IPython.core.display import display
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
    
dropout_value = 0.01
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Prep Layer
        self.convblock01 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value))


        # Layer 1
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1,  bias=False),
            nn.MaxPool2d((2,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value)
            )

        self.residual11 = ResidualBlock(in_channels = 128, out_channels = 128)


        # Layer 2
        self.convblock21 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1,  bias=False),
            nn.MaxPool2d((2,2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value)
            )


        # Layer 3
        self.convblock31 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1,  bias=False),
            nn.MaxPool2d((2,2)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_value)
            )

        self.residual31 = ResidualBlock(in_channels = 512, out_channels = 512)

        self.pool = nn.MaxPool2d((4,4))

        ## Fully Connected Layer
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x1 = self.convblock01(x)
        x2 = self.convblock11(x1)
        x3 = x2 + self.residual11(x2)
        x4 = self.convblock21(x3)
        x5 = self.convblock31(x4)
        x6 = x5 + self.residual31(x5)
        x = self.pool(x6)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = Net()

    def forward(self, x):
        out = self.model.forward(x)
        return F.log_softmax(out, -1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss =nn.CrossEntropyLoss()(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, "multiclass", num_classes = 10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.1,
            weight_decay=1e-4,
        )
        steps_per_epoch = 89600 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                self.hparams.lr,
                epochs=self.trainer.max_epochs,
                pct_start=5/self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
                div_factor=100,
                three_phase=False,
                final_div_factor=100,
                anneal_strategy='linear'
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
