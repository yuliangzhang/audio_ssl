from byol_a2.common import load_yaml_config
from byol_a2.augmentations import PrecomputedNorm
from byol_a2.models import AudioNTT2022, load_pretrained_weights
import nnAudio.features
import torchaudio
import torch


device = torch.device('cuda')
cfg = load_yaml_config('config_v2.yaml')
print(cfg)

# ** Prepare the statistics in advance **
# The followings are mean and standard deviation of the log-mel spectrogram of input audio samples.
# For the SPCV2, this is calculated by using the EVAR terminal output:
#     > byol-a/v2/evar$ python lineareval.py config/byola2.yaml spcv2
#     >   :
#     > using spectrogram norimalization stats: [-9.660292   4.7219563]
stats = [-9.660292, 4.7219563]

# Preprocessor and normalizer.
to_melspec = nnAudio.features.MelSpectrogram(
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
normalizer = PrecomputedNorm(stats)

# Load pretrained weights.
model = AudioNTT2022(n_mels=cfg.n_mels, d=cfg.feature_d)
load_pretrained_weights(model, 'AudioNTT2022-BYOLA-64x96d2048.pth')

# Load your audio file.
wav, sr = torchaudio.load('../work/16k/spcv2/one/00176480_nohash_0.wav') # a sample from SPCV2 for now
assert sr == cfg.sample_rate, "Let's convert the audio sampling rate in advance, or do it here online."

# Convert to a log-mel spectrogram, then normalize.
lms = normalizer((to_melspec(wav) + torch.finfo(torch.float).eps).log())

# Now, convert the audio to the representation.
features = model(lms.unsqueeze(0))
print(features.shape)