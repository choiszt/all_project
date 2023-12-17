### Readme

#### 1- 文件组织结构

```python
├── generate_submission_vision.ipynb #视觉部分的推理和提交代码
├── merge_audio-vision.ipynb #多模态融合部分的训练&模型构建代码
├── multilabel.ipynb #多标签部分的模型构建&训练代码
├── training_vision.ipynb #视觉部分的训练代码
├── 智能网络结题汇报.pptx
├── 智能信息网络实验.pdf
├── bird_clef_audio
├── ├── compute_vad.py #用于对大量音频文件进行语音活动检测，并提取出其中的语音部分，方便后续处理。
├── ├── train.ipynb
├── ├── train.py #audio部分的训练代码
├── ├── vggish_master #vggish模型构建、梅尔图转换、评估矩阵构建部分代码
```

#### 2- 代码运行说明

- 视觉部分的代码训练：

```
运行 training_vision.ipynb
```

- 音频部分的代码训练：

```python
cd bird_clef_audio
python train.py
```

- 多模态融合代码训练：

```
merge_audio-vision.ipynb
```

- 多标签代码训练：

```
multilabel.ipynb
```

- 代码评估和提交：

```
generate_submission_vision.ipynb
```

#### 3- 外部资源

- kaggle平台

  本次实验的大部分训练、推理、评估模块均在kaggle平台上进行。

  本次竞赛的地址为https://www.kaggle.com/competitions/birdclef-2023

- 服务器资源

  由于kaggle难以进行长时间的训练及实验性质的探究，因此部分消融实验考虑在服务器资源上进行部署。

  所使用的四卡服务器配置：gcc version=9.4.0；Ubuntu=9.4.0；python=3.7.7；pytorch=1.6.1

  ![image-20230315143218252](C:\Users\86185\AppData\Roaming\Typora\typora-user-images\image-20230315143218252.png)