from share import *
from pathlib import Path
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pyabdo.utils.file_manager import create_dir

from torch.utils.data import DataLoader
from flyingchairs_dataset import MyFlyingChairs
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


def main():
    parser = ArgumentParser("This script is used to train ControlNet for optical flow")
    parser.add_argument(
        "-c", "--config", type=Path, required=True, help="Path to the yaml config file for the experiment"
    )
    parser.add_argument("-r", "--resume_path", type=Path, required=True, help="Path to the pretrained checkpoint")
    parser.add_argument(
        "-w",
        "--workspace",
        type=Path,
        default="/ibex/user/eldesoa/workspace/ControlNet",
        help="Path to workspace for saving logs and checkpoints",
    )
    parser.add_argument("-p", "--postfix", type=str, default="", help="A postfix to identify the experiemnt")
    args = parser.parse_args()

    create_dir(args.workspace, logger=print)

    # Configs
    batch_size = 16
    logger_freq = 300
    sd_locked = True
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.config).cpu()
    model.load_state_dict(load_state_dict(args.resume_path, location="cpu"))
    model.learning_rate = 1e-5
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Dataset
    dataset = MyFlyingChairs(
        split="train",
        root="/ibex/user/eldesoa/datasets/optical_flow/FlyingChairs_release",
        normalize=True,
    )
    dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)

    # Misc
    logger = ImageLogger(batch_frequency=logger_freq, exp_postfix=args.postfix)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, save_last=True, monitor="train_loss")
    trainer = pl.Trainer(default_root_dir=args.workspace, gpus=1, precision=32, callbacks=[logger, checkpoint_callback])

    # Train!
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
