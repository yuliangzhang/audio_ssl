"""BYOL for Audio: Training.

SYNOPSIS:
    train.py AUDIO_DIR <flags>

FLAGS:
    --config_path=CONFIG_PATH
        Default: 'config.yaml'
    --d=D
        Default: feature_d in the config.yaml
    --epochs=EPOCHS
        Default: epochs in the config.yaml
    --resume=RESUME
        Pathname to the weight file to continue training
        Default: Not specified

Example of training on FSD50K dataset:
    # Preprocess audio files to convert to 16kHz in advance.
    python -m utils.convert_wav /path/to/fsd50k work/16k/fsd50k
    # Run training on dev set for 300 epochs
    python train.py work/16k/fsd50k/FSD50K.dev_audio --epochs=300
"""

from byol_a2.common import (np, Path, torch,
     get_logger, load_yaml_config, seed_everything, get_timestamp, hash_text)
from byol_a2.byol_pytorch_modified import BYOL, loss_fn
from byol_a2.models import AudioNTT2022, TCNModel, load_pretrained_weights
from byol_a2.augmentations import (RandomResizeCrop, MixupBYOLA, RandomLinearFader, NormalizeBatch, PrecomputedNorm)
from byol_a2.dataset import WavDataset
import multiprocessing
import pytorch_lightning as pl
import fire
import logging
import nnAudio.features
from copy import deepcopy
import math
import os

from codecarbon import EmissionsTracker

class AugmentationModule:
    """BYOL-A augmentation module example, the same parameter with the paper."""

    def __init__(self, epoch_samples, log_mixup_exp=True, mixup_ratio=0.2):
        self.train_transform = torch.nn.Sequential(
            MixupBYOLA(ratio=mixup_ratio, log_mixup_exp=log_mixup_exp),
            RandomResizeCrop(virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)),
            RandomLinearFader(),
        )
        logging.info(f'Augmentatoions: {self.train_transform}')

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


class BYOLALearner(pl.LightningModule):
    """BYOL-A learner. Shows batch statistics for each epochs."""

    def __init__(self, cfg, model, tfms, local_tfms, **kwargs):
        super().__init__()
        self.cfg = cfg

        try:
            log_dir = self.logger.log_dir
        except Exception as e:
            log_dir = cfg["log_dir"]
        self.exp_dir = log_dir


        self.global_learner = BYOL(model, image_size=cfg.shape_global, **kwargs)
        tcn = TCNModel(**cfg["tcn_net"])
        self.local_learner = BYOL(tcn, image_size=cfg.shape_local, **kwargs)
        self.lr = cfg.lr
        self.tfms = tfms
        self.local_tfms = local_tfms
        self.post_norm = NormalizeBatch()
        self.to_spec = nnAudio.features.MelSpectrogram(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
            center=True,
            power=2,
            verbose=False,
        )

        hop_length = int(cfg.hop_size * cfg.sample_rate)
        n_frames = math.ceil(cfg.unit_sec / cfg.hop_size)
        clip_audio_win = int(cfg.sample_rate * cfg.win_size)
        last_idx = cfg.sample_rate * cfg.unit_sec


        slice_list = []
        padding_flag = False
        padding_len = last_idx

        for frame_idx in range(n_frames):
            start_idx = frame_idx * hop_length
            end_idx = start_idx + clip_audio_win
            slice_list.append((start_idx, end_idx))

            if end_idx > padding_len:
                padding_flag = True
                padding_len = end_idx

        self.n_frames = n_frames
        # whether the audio will be padded
        self.padding_len = int(padding_len - last_idx)
        self.padding_flag = padding_flag
        # sub-audio segments localization
        self.slice_list = slice_list

        self.linear_combine_loss = torch.nn.MSELoss()
        # self.linear_combine_loss = loss_fn

        self.linear_combine = torch.nn.Linear(cfg["tcn_net"]["num_channels"][-1] * n_frames, 256)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.85)

    # def on_train_start(self) -> None:

        # os.makedirs(os.path.join(self.exp_dir, "training_codecarbon"), exist_ok=True)
        # self.tracker_train = EmissionsTracker("CAFL CONTRASTIVE SED TRAINING",
        #                                       output_dir=os.path.join(self.exp_dir,
        #                                                               "training_codecarbon"))
        # self.tracker_train.start()

    def forward(self, images1, images2):
        return self.learner(images1, images2)

    def training_step(self, wavs, batch_idx):
        def to_np(A): return [a.cpu().numpy() for a in A]
        # Convert raw audio into a log-mel spectrogram and pre-normalize it.
        self.to_spec.to(self.device, non_blocking=True)
        self.global_learner.to(self.device, non_blocking=True)
        self.local_learner.to(self.device, non_blocking=True)
        self.linear_combine.to(self.device, non_blocking=True)
        self.linear_combine_loss.to(self.device, non_blocking=True)

        # split wavs into sub wavs
        # [2] get local embeddings
        if self.padding_flag:
            padded_audios = torch.nn.functional.pad(
                wavs, (0, self.padding_len), mode="constant")
        else:
            padded_audios = wavs

        # combine all the splitted sub-audio togethers
        split_audio_list = [padded_audios[:, start_idx:end_idx] for start_idx, end_idx in self.slice_list]
        split_audio_combine = torch.cat(split_audio_list, dim=0)

        lms_batch = (self.to_spec(wavs) + torch.finfo().eps).log().unsqueeze(1)
        sub_lms_batch = (self.to_spec(split_audio_combine) + torch.finfo().eps).log().unsqueeze(1)

        lms_batch = self.pre_norm(lms_batch)
        sub_lms_batch = self.pre_norm(sub_lms_batch)

        # Create two augmented views.
        images1, images2 = [], []
        for lms in lms_batch:
            img1, img2 = self.tfms(lms)
            images1.append(img1), images2.append(img2)
        images1 = torch.stack(images1)
        images2 = torch.stack(images2)
        paired_inputs = (images1, images2)

        # Create two augmented sub views.
        sub_images1, sub_images2 = [], []
        for sub_lms in sub_lms_batch:
            img1, img2 = self.local_tfms(sub_lms)
            sub_images1.append(img1), sub_images2.append(img2)

        sub_images1 = torch.stack(sub_images1)
        sub_images2 = torch.stack(sub_images2)
        sub_paired_inputs = (sub_images1, sub_images2)

        # Form a batch and post-normalize it.
        bs = paired_inputs[0].shape[0]
        paired_inputs = torch.cat(paired_inputs) # [(B,1,T,F), (B,1,T,F)] -> (2*B,1,T,F)
        mb, sb = to_np((paired_inputs.mean(), paired_inputs.std()))
        paired_inputs = self.post_norm(paired_inputs)
        ma, sa = to_np((paired_inputs.mean(), paired_inputs.std()))

        # split the mel frames into sub-mel frames
        global_frame1 = paired_inputs[:bs]
        global_frame2 = paired_inputs[bs:]

        # Form a batch and post-normalize it.
        sub_bs = sub_paired_inputs[0].shape[0]
        sub_paired_inputs = torch.cat(sub_paired_inputs)  # [(B,1,T,F), (B,1,T,F)] -> (2*B,1,T,F)
        sub_paired_inputs = self.post_norm(sub_paired_inputs)

        # split the mel frames into sub-mel frames
        local_frame1 = sub_paired_inputs[:sub_bs]
        local_frame2 = sub_paired_inputs[sub_bs:]


        # Forward to get a loss.
        loss_global, global_proj1, global_proj2 = self.global_learner(global_frame1, global_frame2)
        loss_local, local_proj1, local_proj2 = self.local_learner(local_frame1, local_frame2)

        local_proj1_list = torch.split(local_proj1, split_size_or_sections=bs, dim=0)
        local_proj2_list = torch.split(local_proj2, split_size_or_sections=bs, dim=0)

        local_proj1 = torch.stack(local_proj1_list, dim=1).to(torch.float32)
        local_proj2 = torch.stack(local_proj2_list, dim=1).to(torch.float32)

        local_proj1_combine = torch.unbind(local_proj1, dim=1)
        local_proj2_combine = torch.unbind(local_proj2, dim=1)

        local_proj1_combine = torch.cat(local_proj1_combine, dim=1)
        local_proj2_combine = torch.cat(local_proj2_combine, dim=1)

        final_local_proj1 = self.linear_combine(local_proj1_combine)
        final_local_proj2 = self.linear_combine(local_proj2_combine)


        combine_loss1 = self.linear_combine_loss(global_proj1, final_local_proj1)
        combine_loss2 = self.linear_combine_loss(global_proj2, final_local_proj2)

        combine_loss = (combine_loss1 + combine_loss2) / 2.0
        # combine_loss = combine_loss.mean()
        total_loss = loss_global + loss_local + combine_loss * self.cfg.combine_loss_weight

        self.log("total_loss", total_loss)
        self.log("global_loss", loss_global)
        self.log("local_loss", loss_local)
        self.log("combine_loss", combine_loss)
        self.log("learning_rate", self.optimizer.param_groups[-1]["lr"], prog_bar=True)
        # self.log("combine_loss2", combine_loss2)

        for k, v in {'mb': mb, 'sb': sb, 'ma': ma, 'sa': sa}.items():
            self.log(k, float(v), prog_bar=True, on_step=False, on_epoch=True)

        print("\ntotal_loss=", total_loss.item(), " global_loss=", loss_global.item(), " local_loss=", loss_local.item(),
              " combine_loss=", combine_loss.item())
        return total_loss

    def configure_optimizers(self):

        return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler": self.scheduler}}

    def on_before_zero_grad(self, _):
        self.global_learner.update_moving_average()
        self.local_learner.update_moving_average()

    def calc_norm_stats(self, data_loader, n_stats=10000, device='cuda'):
        # Calculate normalization statistics from the training dataset.
        n_stats = min(n_stats, len(data_loader.dataset))
        logging.info(f'Calculating mean/std using random {n_stats} samples from population {len(data_loader.dataset)} samples...')
        self.to_spec.to(device)
        X = []
        for wavs in data_loader:
            lms_batch = (self.to_spec(wavs.to(device)) + torch.finfo().eps).log().unsqueeze(1)
            X.extend([x for x in lms_batch.detach().cpu().numpy()])
            if len(X) >= n_stats: break
        X = np.stack(X)
        norm_stats = np.array([X.mean(), X.std()])
        logging.info(f'  ==> mean/std: {norm_stats}, {norm_stats.shape} <- {X.shape}')
        self.pre_norm = PrecomputedNorm(norm_stats)
        return norm_stats


def complete_cfg(cfg):
    # Set ID.
    cfg.id = (f'AudioNTT2022-BYOLA-{cfg.shape_global[0]}x{cfg.shape_global[1]}d{cfg.feature_d}-{get_timestamp()}'
              f'-e{cfg.epochs}b{cfg.bs}l{str(cfg.lr)[2:]}r{cfg.seed}-{hash_text(str(cfg), L=8)}')
    return cfg


def main(config_path='config_calf_tcn_split.yaml', d=None, epochs=None, resume=None) -> None:
    audio_dir_list = ["../work/16k/fsd50k_dev"]
    cfg = load_yaml_config(config_path)
    # Override configs
    cfg.feature_d = d or cfg.feature_d
    cfg.epochs = epochs or cfg.epochs
    cfg.resume = resume or cfg.resume
    cfg.unit_samples = int(cfg.sample_rate * cfg.unit_sec)
    complete_cfg(cfg)
    # Essentials
    get_logger(__name__)
    logging.info(cfg)
    seed_everything(cfg.seed)
    # Data preparation
    files = []
    for audio_dir in audio_dir_list:
        files.extend(Path(audio_dir).glob('*.wav'))
        # files = sorted(Path(audio_dir).glob('*.wav'))

    files = sorted(files)
    tfms = AugmentationModule(epoch_samples=2 * len(files))
    local_tfms = AugmentationModule(epoch_samples=2 * len(files))
    ds = WavDataset(cfg, files, labels=None, tfms=None, random_crop=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.bs,
                num_workers=multiprocessing.cpu_count(),
                pin_memory=True, shuffle=True,)
    logging.info(f'Dataset: {len(files)} .wav files from {audio_dir}')
    # Training preparation
    logging.info(f'Training {cfg.id}...')
    # Model
    model = AudioNTT2022(n_mels=cfg.n_mels, d=cfg.feature_d)
    if cfg.resume is not None:
        load_pretrained_weights(model, cfg.resume)
    # Training
    learner = BYOLALearner(cfg, model, tfms=tfms, local_tfms=local_tfms,
        hidden_layer=-1,
        projection_size=cfg.proj_size,
        projection_hidden_size=cfg.proj_dim,
        moving_average_decay=cfg.ema_decay,
    )
    learner.calc_norm_stats(dl, n_stats=10000)
    # trainer = pl.Trainer(gpus=cfg.gpus, max_epochs=cfg.epochs, weights_summary=None, accelerator="ddp")
    # checkpoint_resume = "/home/ubuntu/work_dir/audio_ssl/v2/lightning_logs/version_3/checkpoints/epoch=41-step=71693.ckpt"
    trainer = pl.Trainer(gpus=cfg.gpus,
                         max_epochs=cfg.epochs,
                         # resume_from_checkpoint=checkpoint_resume
                         )
    trainer.fit(learner, dl)
    if trainer.interrupted:
        logging.info('Terminated.')
        exit(0)
    # Saving trained weight.
    to_file = Path(cfg.checkpoint_folder)/(cfg.id+'.pth')
    to_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), to_file)
    logging.info(f'Saved weight as {to_file}')


if __name__ == '__main__':
    fire.Fire(main)
    # ckpt_file = "/home/ubuntu/work_dir/audio_ssl/v2/lightning_logs/version_4/checkpoints/epoch=130-step=223616.ckpt"
    # state_dict = torch.load(ckpt_file)
    # if 'state_dict' in state_dict:
    #     state_dict = state_dict['state_dict']
    # if 'model' in state_dict:
    #     state_dict = state_dict['model']
    #
    # print(state_dict)



