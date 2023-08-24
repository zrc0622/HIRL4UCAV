
# Reinforcement Learning

* Install Latest Version of Sandbox
  
  * Install latest version of Sandbox. Download from "[Releases](https://github.com/harfang3d/dogfight-sandbox-hg2/releases/tag/v1.3.0)" and you can run directly.

* Install Packages
  * Use pip to install the requirements.

    ~~~bash
    pip install -r requirements.txt
    ~~~
* Change IP and PORT
  * Change IP and PORT values from Train.py file
* Run
  * Run Train.py file


# Get Expert Date

## Rule-based
* run demo.py file which will then generate a pkl file named "expert_data"
* video</br>[![rule-based expert demo](https://github.com/zrc0622/harfang-sandbox/blob/master/pictures/1.jpg =400x)](https://www.youtube.com/watch?v=i6DAneyneh8 "rule-based expert demo")
## AI-based
* run demo_AI.py file which will then generate a pkl file named "expert_data_new"
* video</br>[![AI-based expert demo](https://github.com/zrc0622/harfang-sandbox/blob/master/pictures/3.jpg =400x)](https://www.youtube.com/watch?v=uQKoI0rQC2k "AI-based expert demo")
# Data Description
## expert_data_new.pkl
* 基于AI的agent生成的“追逐敌机并发射导弹将敌机击落”的轨迹，可用于imitation learning
* 轨迹数量：50
* 轨迹长度：107859
* 状态空间：17维（相对位置（3），agent的欧拉角（3），agent的航向角，敌机的航向角，敌机的俯仰角，敌机的旋转角，目标角度，敌机健康值，是否锁敌，导弹状态（4））
* 动作空间：4维

  
