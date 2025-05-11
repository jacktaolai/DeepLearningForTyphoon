

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue.svg"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg"/>
  <img src="https://img.shields.io/badge/Dataset-DigitalTyphoon-orange.svg"/>
</p>

<h1 align="center">🌪️ DeepLearningForTyphoon</h1>
<div align="center">
  <img src="assets\台风深度学习横幅.png" alt="typhoon icon"/>
</div>

<p align="center">台风智能识别与预测的深度学习项目 | <i>专为卫星云图打造的AI套件</i></p>



---

## 📑 项目简介

> ✨ **DeepLearningForTyphoon** 为台风智能识别、分型与路径/强度预测提供了完善的开源深度学习解决方案，专注于处理卫星遥感影像以及关联的元数据。

- 包含 **ResNet**/LSTM 等多种模型实现，适用于分类、回归等核心任务。
- 灵活适配日本国立情报学研究所（NII）【Digital Typhoon】数据集。
- 适用于学术研究、科研竞赛及行业工程落地。

---

## 🗂️ 数据组织结构

```
项目根目录/
├── images/                 # 卫星云图目录
│   ├── 200501/             # 台风编号文件夹
│   ├── 200502/
│   └── ...
├── metadata/               # 元数据目录
│   ├── 200501.csv
│   └── ...
└── metadata.json           # 全局元数据
```
> 💡 数据请严格按照上述目录组织，并在 `config` 内指定实际路径！

---

## 🌐 数据集介绍

- **[Digital Typhoon open dataset (NII)](https://agora.ex.nii.ac.jp/digital-typhoon/dataset/)**
    - **WP**: 西北太平洋台风, 1978-2023, *56GB*
    - **AU**: 澳大利亚地区台风, 1979-2024, *21GB*


---

<details>
<summary><b>📦 任务与模型列表</b></summary>

| 任务名称                      | 类型     | 输入格式                | 模型结构                     |
|-----------------------------|----------|------------------------|----------------------------|
| `ClassicalModel`            | 分类     | 随机单张台风云图        | 标准CNN等分类架构           |
| `DovarkResnet`              | 分类     | 随机单张台风云图        | 自定义改进版ResNet18        |
| `ResnetLSTMClassification`  | 分类     | 图片滑窗（时序序列）    | ResNet18 + LSTM            |
| `ResnetLSTMRegression`      | 回归预测 | 图片滑窗（时序序列）    | ResNet18 + LSTM            |

</details>

---

## ⚒️ 环境安装及依赖

<details>
<summary>🛠️ <b>依赖环境说明（点击展开）</b></summary>

### 推荐：conda/miniconda 管理python环境

- 通用环境（支持所有任务）：

  ```bash
  conda env create -f environment.yml
  conda activate typhoon
  ```

- 针对Autodl算力平台与LSTM任务（推荐自动求解LSTM模型环境）:
  ```bash
  conda env create -f autodl_environment.yml
  conda activate typhoon_autodl
  ```

#### 🌏 数据加载依赖 [pyphoon2]

- **方式1：本项目自带**
    ```bash
    cd pyphoon2
    pip3 install .
    ```
- **方式2：官方仓库**
    ```bash
    git clone https://github.com/kitamoto-lab/pyphoon2
    cd pyphoon2
    pip3 install .
    ```
- **卸载**
    ```bash
    pip3 uninstall pyphoon2
    ```

</details>

---

## 📝 快速开始

1. **配置数据及`config`路径参数**
2. **激活环境**
3. **执行各子任务脚本**

_示例：_
```bash
python train_resnetlstm_classification.py
```

> ⚠️ 每次变更数据集范围，请及时同步/检查**根目录**`metadata.json` 内容，即使数据集存在，但根目录`metadata.json`不包括该台风，模型也不会使用该部分数据。同理，你也可以通过修改`metadata.json`来实现对需要的数据集的控制。

---

<details>
<summary><b>📚 参考资源与官方资料</b></summary>

- [Digital Typhoon数据集主页](https://agora.ex.nii.ac.jp/digital-typhoon/dataset/)
- [pyphoon2官方库](https://github.com/kitamoto-lab/pyphoon2)

</details>

---

## 🏆 贡献&鸣谢

- 欢迎提交 Issue / Pull Request，助力台风AI能力提升！
- 感谢 NII（日本国立情报学研究所）的开放数据支持。

---

<div align="center">
  <img src="https://img.icons8.com/color/48/000000/tornado.png"/>
  <br>
  <b>如在学术/生产环境使用本项目，请注明引用</b>
  <br>
  <i>— 万千台风，从“云”出发 —</i>
</div>

---

✨ **Any questions and suggestions are welcome!**
🌊 让AI科技术语融化在风雨云端，让世界预知风暴！

---
