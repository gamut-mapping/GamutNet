import time
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets.data_modules import WideGamutNetDataModule
from models import WideGamutNetPL


def main(args):
    # checkpointing
    checkpoint_callback = ModelCheckpoint(save_top_k=-1,  # always save all the checkpoints
                                          verbose=True, )
    trainer = Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    model = WideGamutNetPL(hparams=args)  # pick model
    datamodule = WideGamutNetDataModule(hparams=args)  # pick datamodule
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path=None)  # use the latest weights (because we're saving all the checkpoints, not the best one)


if __name__ == '__main__':
    start_time = time.time()

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = WideGamutNetPL.add_model_specific_args(parser)  # model specific arguments
    parser = WideGamutNetDataModule.add_datamodule_specific_args(parser)  # datamodule specific arguments
    main(parser.parse_args())  # parse args and start training

    end_time = time.time()
    duration = end_time - start_time
    duration = round(duration/3600, 2)
    print(f'---- FINISHED in {duration} hours ----')
