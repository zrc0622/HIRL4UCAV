[**ENGLISH**](README.md) | [**中文**](README_CN.md)

<h1 align='center'> Highly Imitative Reinforcement Learning for UCAV </h1>

## Result Charts

<p float="left">
  <img src="results/vr_vsr.png" width="" />
  <!-- <img src="results/vsr.png" width="48%" /> -->
</p>


## Models and Results
The best models trained by BC、TD3、SAC、E-SAC、HIRL (our method) are stored in the `./results` folder. The validation results of the models are as follows (validation results without random initialization and with random initialization are obtained by running 50 episodes with 5 different random seeds, '±' indicates standard deviation; the number of hits and launches are obtained by running 10 episodes; the best results are highlighted in bold).
### Validation Results without Random Initialization
| **Methods**              | **Shoot-down Success Rate**        | **Hit Success Rate**        | **Rewards**           |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| HIRL (adaptive)  | **100.0% ± 0.0%**     | **100.0% ± 0.0%**     | **-680.8 ± 6.7**      |
| HIRL (linear)    | **100.0% ± 0.0%**     | **100.0% ± 0.0%**    | -953.9 ± 13.8     |
| TD3             | 0.0% ± 0.0%       | 0.0% ± 0.0%       | -4707.2 ± 0.0     |
| E-SAC           | **100.0% ± 0.0%**     | **100.0% ± 0.0%**     | -1431.2 ± 0.2     |
| SAC             | **100.0% ± 0.0%**     | 0.0% ± 0.0%       | -2985.7 ± 0.0     |
| BC              | 62.8% ± 1.0%      | 62.8% ± 1.0%      | -12228.3 ± 880.2  |
### Validation Results with Random Initialization
| **Methods**             | **Shoot-down Success Rate**       | **Hit Success Rate**       | **Rewards**               |
|:-----------------:|:----------------:|:----------------:|:---------------------:|
| HIRL (adaptive)  | **98.0% ± 1.3%**     | **98.0% ± 1.3%**     | **-1436.0 ± 238.9**       |
| HIRL (linear)    | 86.0% ± 5.4%     | 86.0% ± 5.4%     | -5800.8 ± 1420.3      |
| TD3             | 0.0% ± 0.0%      | 0.0% ± 0.0%      | -5720.9 ± 715.8       |
| E-SAC           | 90.0% ± 2.8%     | 90.0% ± 2.8%     | -3722.2 ± 395.5       |
| SAC             | 44.0% ± 3.3%     | 0.0% ± 0.0%      | -8318.1 ± 822.8       |
| BC              | 22.4% ± 3.2%     | 22.4% ± 3.2%     | -20504.7 ± 1156.3     |

### Launch Efficiency Results
| **Methods**             | **Hits / Launches** |
|:-----------------:|:-----------------:|
| HIRL (adaptive)  | **100.0%**          |
| HIRL (linear)    | **100.0%**          |
| E-SAC           | 11.4%           |
| BC              | 92.3%           |



## Policy Display
> Display the effectiveness of the policy trained by the HIRL.

### One Missile

![image](videos/single.gif)

### Infinite Missiles

![image](videos/multi.gif)

## Getting Started

### Installation Requirements
1. It is recommended to use a computer with Windows operating system (we have tried using Linux, but it seems that Harfang3D is not compatible).
2. Install `Harfang3D sandbox` from the [release](https://github.com/harfang3d/dogfight-sandbox-hg2/releases/tag/v1.3.0) or [source](https://github.com/harfang3d/dogfight-sandbox-hg2). It is recommended to install from [source](https://github.com/harfang3d/dogfight-sandbox-hg2) for more flexibility, such as customizing the network port of the environment.
3. Install the dependencies required for this code.
    ```
    conda env create -f environment.yaml
    ```
### Training
1. In the `Harfang3D sandbox` folder, use the following command to open `Harfang3D sandbox`. You can specify the port number with `network_port`. After opening, you need to manually enter the **network mode**.
    ```bash
    cd source
    python main.py network_port 12345
    ```
2. In the `HIRL4UCAV` folder, use the following command to start training (note to modify the IP number in the `train_all.py`; use `--render` to enable training rendering, and use `--plot` to draw visualization results).
    ```bash
    # HIRL (adaptive)
    python train_all.py --agent ROT --port 12345 --type soft --model_name srot
    ```
    ```bash
    # HIRL (linear)
    python train_all.py --agent ROT --port 12345 --type linear --bc_weight 1 --model_name lrot
    ```
    ```bash
    # HIRL (fixed)
    python train_all.py --agent ROT --port 12345 --type fixed --bc_weight 0.5 --model_name frot
    ```
    ```bash
    # TD3
    python train_all.py --agent TD3 --port 12345 --model_name td3
    ```
    ```bash
    # BC
    python train_all.py --agent BC --port 12345 --model_name bc
    ```
    ```bash
    # SAC
    python train_sac.py --type sac --port 12345 --model_name sac
    ```
    ```bash
    # E-SAC
    python train_sac.py --type esac --port 12345 --model_name esac
    ```
### Validation
1. In the `Harfang3D sandbox` folder, use the following command to open `Harfang3D sandbox`. You can specify the port number with `network_port`. After opening, you need to manually enter the **network mode**.
    ```bash
    cd source
    python main.py network_port 12345
    ```
2. To test the BC, TD3, and HIRL models, use the following command in the `HIRL4UCAV` folder (note to modify the IP number and the model name in the `train_all.py` (only the name before 'xxx_Harfang_GYM' is needed); use `--render` to enable test rendering).
    ```bash
    # Sucess Rate Validation
    # Add '--test --test_mode n' to the end of the corresponding training command. 'test mode 1' is the random initialization mode, 'test mode 2' is the infinite missiles mode, and 'test mode 3' is the original environment
    # Here's an example
    python train_all.py --agent ROT --port 12345 --type soft --model_name srot --test --test_mode 1 --seed 1
    ```
    ```bash
    # Reward Validation
    # Add '--test --test_mode n' to the end of the corresponding training command. 'test mode 4' is the random initialization mode, and 'test mode 5' is the original environment
    # Here's an example
    python train_all.py --agent ROT --port 12345 --type soft --model_name srot --test --test_mode 4 --seed 1
    ```
3. To test the SAC and E-SAC models, use the following command in the `HIRL4UCAV` folder (types of test mode are as described above).
    ```
    python validate_sac.py --test_mode 1 --port 12345 --seed 1 
    ```

## Citation
```
@misc{li2024imitative,
    title={An Imitative Reinforcement Learning Framework for Autonomous Dogfight}, 
    author={Siyuan Li and Rongchang Zuo and Peng Liu and Yingnan Zhao},
    year={2024},
    eprint={2406.11562},
    archivePrefix={arXiv}
}
```