
# Diff-SVC
Singing Voice Conversion via diffusion model

## This repository is a refactored version of diff-svc fork, with new features such as multi-speaker support, auxiliary scripts, and new Hubert. Please evaluate and assume the risks on your own.
> We recommend using the stable version: [Diff-SVC](https://github.com/prophesier/diff-svc)
> 
> The project tutorial can be found in the doc folder. Please do not ask questions about this modified version in the original project channel or Discord.
> 
> Under the same parameters, the number of training steps required for the Chinese Hubert is approximately 1.5 to 2 times that of Soft Hubert. It is not recommended for beginners to use.

## Changes Log
> 2023.03.09
> 
> Optimized the speed of nsf-hifigan @diffsinger
> 
> 2023.02.18
> 
> Updated config parameters, added flask_api multi-speaker model, and removed midi a mode diffsinger nesting support @小狼
> 
> 2023.01.20
> 
> Refactored directory, streamlined code, and removed multiple inheritance @小狼
> 
> 2023.01.18
> 
> Changed configuration file to cascading, only need to modify config_nsf, config_ms (choose one) for preprocessing @小狼
> 
> 2023.01.16
> 
> Added multi-speaker support (config_ms.yaml), preprocessing code referenced diffsinger modified by @小狼
> 
> 2023.01.09
> 
> Added select.py to filter the pitch range of the dataset (when the amount of data is sufficient, remove the duplicate pitch range to speed up the convergence of high and low pitches)
> 
> Removed dependencies on 24k pe and hifigan, deleted pitch cwt mode, and reused preprocessing code for inference @小狼
> 
> 2023.01.07
> 
> Added f0_static parameter for statistical pitch range, and added adaptive pitch shift function (requires f0_static, old model config can use data_static to add this parameter) @小狼
> 
> 2023.01.05
> 
> Cancelled support for 24k sampling rate and pe, reduced some parameters, added specialized tutorials to the documentation; batch.py supports both specialized and nesting mode export;
> 
> pre_hubert is a step-by-step preprocessing (used for preprocessing with 4g or less memory); data_static is for dataset pitch range statistics (for reference only); > > Chinese Hubert requires fairseq dependency, please install it yourself @小狼
> 
> 2023.01.01
> 
> Updated slicer v2, removed slicer cache, simplified some infer processes; removed vec support, added Chinese Hubert (only base model, around 1.1g) @小狼
> 
> 2022.12.17
> 
> Added pre_check to detect environment and data @深夜诗人; improved simplify model @九尾玄狐; supervised code @小狼
> 
> 2022.12.16
> 
> Fixed the problem of repeatedly loading the hubert model during inference @小狼
> 
> 2022.12.04
> 
> Opened application for 44.1kHz vocoder and officially provided support for 44.1kHz.
> 
> 2022.11.28
> 
> An option, no_fs2, has been added by default which can optimize some networks, improve training speed, reduce model size, and be effective for new models trained in the future.
> 
> 2022.11.23
> 
> A major bug has been fixed that could potentially convert the original gt audio used for inference to a sampling rate of 22.05kHz. We apologize for any inconvenience caused and kindly ask that you check your test audio and use the updated code.
> 
> 2022.11.22
> 
> Many bugs have been fixed, including several major ones that have a significant impact on inference performance.
> 
> 2022.11.20
> 
> Added support for most formats of input and output during inference, eliminating the need for manual conversion using other software.
> 
> 2022.11.13
> 
> Corrected the epoch/steps display issue when reading models after interruption, added disk cache for f0 processing, and added support files for real-time pitch-shifting inference.
> 
> 2022.11.11
> 
> Corrected the duration error during slicing, added support for 44.1kHz, and added support for contentvec.
> 
> 2022.11.04
> 
> Added the feature to save mel-spectrograms.
> 
> 2022.11.02
> 
> Integrated the new vocoder code and updated the parselmouth algorithm.
> 
> 2022.10.29
> 
> Organized the inference section and added the feature for automatic slicing of long audio files.
> 
> 2022.10.28
> 
> Migrated the hubert onnx inference to torch inference and reorganized the inference logic. If you have previously downloaded the onnx hubert model, please download and replace it with the pt model. The configuration file does not need to be changed. Currently, direct GPU inference and preprocessing on a 1060 6G GPU is possible. For more details, please refer to the documentation.
> 
> 2022.10.27
> 
> Updated dependency files and removed redundant dependencies.
> 
> 2022.10.27
> 
> Fixed a severe bug that caused hubert to use CPU inference on a GPU server, slowing it down by 3-5 times. This issue affects preprocessing and inference, but not training.
> 
> 2022.10.26
> 
> Fixed the issue of preprocessed data on Windows not being usable on Linux and updated some documentation.
> 
> 2022.10.25
> 
> Wrote detailed documentation for inference and training, modified and integrated some code, and added support for ogg audio format (no need to distinguish between ogg and wav, can be used directly).
> 
> 2022.10.24
> 
> Supported training on custom datasets and simplified the code.
> 
> 2022.10.22
> 
> Completed training on the opencpop dataset and created a repository.
## 注意事项 /Notes：

> 本项目是基于学术交流目的建立，并非为生产环境准备，不对由此项目模型产生的任何声音的版权问题负责。
>
> 如将本仓库代码二次分发，或将由此项目产出的任何结果公开发表 (包括但不限于视频网站投稿)，请注明原作者及代码来源 (此仓库)。
>
> 如果将此项目用于任何其他企划，请提前联系并告知本仓库作者，十分感谢。

> This project is established for academic exchange purposes and is not intended for production environments. We are not
>
> responsible for any copyright issues arising from the sound produced by this project's model. 
>
> If you redistribute the code in this repository or publicly publish any results produced by this project (including but not limited to video website submissions), please indicate the original author and source code (this repository). 
>
> If you use this project for any other plans, please contact and inform the author of this repository in advance. Thank you very much.

## 推理 /Inference：

参考 `infer.py` 进行修改

## 预处理 /PreProcessing:

```sh
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python preprocessing/svc_binarizer.py --config configs/config_nsf.yaml
```

## 训练 /Training:

```sh
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/config_nsf.yaml --exp_name <your project name> --reset 
```
> Links:
>
> 详细训练过程和各种参数介绍请查看 [推理与训练说明](./doc/train_and_inference.markdown) 
>
> [中文 hubert 与特化教程](./doc/advanced_skills.markdown) 


## 学术 / Acknowledgements

项目基于 [diffsinger](https://github.com/MoonInTheRiver/DiffSinger)、[diffsinger (openvpi 维护版)](https://github.com/openvpi/DiffSinger)、[soft-vc](https://github.com/bshall/soft-vc)
开发.

同时也十分感谢 openvpi 成员在开发训练过程中给予的帮助。

This project is based
on [diffsinger](https://github.com/MoonInTheRiver/DiffSinger), [diffsinger (openvpi maintenance version)](https://github.com/openvpi/DiffSinger),
and [soft-vc](https://github.com/bshall/soft-vc). We would also like to thank the openvpi members for their help during
the development and training process. 

> 注意：此项目与同名论文 [DiffSVC](https://arxiv.org/abs/2105.13871) 无任何联系，请勿混淆！

> Note: This project has no connection with the paper of the same name [DiffSVC](https://arxiv.org/abs/2105.13871),
> please
> do not confuse them!

## 工具 / Tools

音频切片参考 [audio-slicer](https://github.com/openvpi/audio-slicer)

Audio Slice Reference [audio-slicer](https://github.com/openvpi/audio-slicer)
