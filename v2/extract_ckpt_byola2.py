import torch
from byol_a2.models import AudioNTT2022
from byol_a2.common import load_yaml_config
if __name__ == '__main__':
    ckpt_file = "/home/ubuntu/work_dir/audio_ssl/v2/lightning_logs/version_3/checkpoints/epoch_8.ckpt"
    # config_path = 'config_calf.yaml'
    # model_key = ""
    #
    # cfg = load_yaml_config(config_path)

    # model = AudioNTT2022(n_mels=cfg.n_mels, d=cfg.feature_d)
    state_dict = torch.load(ckpt_file)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if 'model' in state_dict:
        state_dict = state_dict['model']

    # children = sorted([n + '.' for n, _ in model.named_children()])

    # 'model.xxx' -> 'xxx"
    # weights = {}
    # for k in state_dict:
    #     if k.startswith(model_key+'.'):
    #         weights[k[len(model_key)+1:]] = state_dict[k]
    # state_dict = weights
    #
    # # model's parameter only
    # def find_model_prm(k):
    #     for name in children:
    #         if name in k: # ex) "conv_block1" in "model.conv_block1.conv1.weight"
    #             return k
    #     return None
    #
    # weights = {}
    # for k in state_dict:
    #     if find_model_prm(k) is None: continue
    #     weights[k] = state_dict[k]
    #
    # print(list(weights.keys()))
    # print(str(model.load_state_dict(weights, strict=True)))
    # return sorted(list(weights.keys()))