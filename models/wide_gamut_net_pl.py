from argparse import ArgumentParser
import torch
from torch.nn import functional as F
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule
from models.networks.wide_gamut_net import WideGamutNet, SmallWideGamutNet, TinyWideGamutNet


class WideGamutNetPL(LightningModule):

    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(hparams)
        if self.hparams.model_size == 'default':  # RF=82
            self.net = WideGamutNet(self.hparams.input_channels,
                                    using_residual=self.hparams.using_residual,
                                    limiting_output_range=self.hparams.limiting_output_range)
        elif self.hparams.model_size == 'small':  # RF=68
            self.net = SmallWideGamutNet(self.hparams.input_channels,
                                         using_residual=self.hparams.using_residual,
                                         limiting_output_range=self.hparams.limiting_output_range)
        elif self.hparams.model_size == 'tiny':  # RF=32
            self.net = TinyWideGamutNet(self.hparams.input_channels,
                                        using_residual=self.hparams.using_residual,
                                        limiting_output_range=self.hparams.limiting_output_range)
        else:
            raise Exception("Sorry, there is no model like that!")

        # the input array is in shape of (batch_size, input_channels, patch_height, patch_size)
        self.example_input_array = torch.rand(self.hparams.batch_size,
                                              self.hparams.input_channels,
                                              *self.hparams.patch_size,
                                              device=self.device)

    def _loss(self, batch):
        input_patches, target_patches, m2o_mask_patches = batch
        predicted_patches = self(input_patches.float())  # get predictions
        target_patches = target_patches.float()  # get targets
        y_hat = predicted_patches[m2o_mask_patches]  # consider only many-to-one pixels
        y = target_patches[m2o_mask_patches]  # consider only many-to-one pixels
        return F.l1_loss(y_hat, y)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group("WideGamutNetPL", "Arguments for WideGamutNetPL")

        # for model configuration
        model_parser.add_argument('--model_size', type=str, default='default', choices=['default', 'small', 'tiny'])
        model_parser.add_argument('--input_channels', type=int, default=4, choices=[3, 4, 6])

        # for optimizer configuration
        model_parser.add_argument('--learning_rate', type=float, default=0.0001)

        model_parser.add_argument('--using_residual', type=bool, default=True)
        model_parser.add_argument('--limiting_output_range', type=bool, default=True)

        return parent_parser
