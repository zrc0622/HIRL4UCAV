[**ENGLISH**](../README.md) | [**中文**](README_CN.md)

<h1 align='center'> 基于高模仿性强化学习的无人机空战 </h1>

本项目是论文[《An Imitative Reinforcement Learning Framework for Pursuit-Lock-Launch Missions》](https://arxiv.org/abs/2406.11562)的实现。专家数据集、训练好的模型以及策略的演示视频可在[Google Drive](https://drive.google.com/drive/folders/1lAllxmsy0MhW714ZmT8fb0MkdJktUxzJ?usp=sharing)获取。

## 配置

### 环境依赖

请按照[官方说明](https://github.com/harfang3d/dogfight-sandbox-hg2)安装 Dogfight Sandbox。

<!-- Alternatively, installation can also be done directly from one of the following links: [Link 1](https://github.com/harfang3d/dogfight-sandbox-hg2/releases/download/v1.3.1/dogfight-sandbox-hg2.zip) or [Link 2](https://drive.google.com/file/d/1FihtrwnwGt0FXaVlGS4881yN3oYpbdlw/view?usp=drive_link). -->

### HIRL

```bash
conda create -n hirl python=3.8
conda activate hirl
git clone https://github.com/zrc0622/HIRL4UCAV.git
cd HIRL4UCAV
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -e .
```

## 使用说明

### 准备工作
从[Google Drive](https://drive.google.com/drive/folders/1lAllxmsy0MhW714ZmT8fb0MkdJktUxzJ?usp=sharing)下载`expert_data`和`bc_actor`文件夹并放置于项目根目录，然后更新`local_config.yaml`文件中的IP地址配置。

### 运行实验
完成准备工作后，请按照以下步骤运行实验：

1. **启动环境：**运行 Dogfight Sandbox 并选择 NETWORK MODE 模式

2. **开始训练：**环境启动后，使用以下命令运行实验：
    ```bash
    python hirl/train_all.py --port=<ENV_PORT> --env=<ENV_TYPE> --random --agent=HIRL --type=<HIRL_TYPE> --model_name=<MODEL_NAME>
    ```

请根据实际需求替换相应参数。

<!-- - `<ENV_PORT>`: The port number for the training environment (e.g., 12345).
- `<ENV_TYPE>`: The type of training environment (e.g., "straight_line", "serpentine", "circular").
- `<HIRL_TYPE>`: The variant of the HIRL algorithm (e.g., "soft", "linear", "fixed").
- `<MODEL_NAME>`: The name of the trained model to be saved (e.g., "HIRL_soft"). -->

## 性能表现

### 对比实验结果

<div align="center">
  <img src="./fig1.png" width="100%"/>
</div>

### 策略轨迹可视化

<div align="center">
  <img src="./fig2.png" width="100%"/>
</div>

## 引用
```bibtex
@misc{li2025imitativereinforcementlearningframework,
      title={An Imitative Reinforcement Learning Framework for Pursuit-Lock-Launch Missions}, 
      author={Siyuan Li and Rongchang Zuo and Bofei Liu and Yaoyu He and Peng Liu and Yingnan Zhao},
      year={2025},
      eprint={2406.11562},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.11562}, 
}
```