# CNN图像风格迁移项目

## 项目概述

这是一个基于深度学习的图像风格迁移项目，实现了多种风格迁移算法，包括传统的基于优化的方法和快速前馈风格迁移方法。项目提供了一个完整的Web界面，允许用户上传内容图片和风格图片，调整参数并实时查看训练过程和结果。

## 项目特点

- **多种实现方式**：支持Keras和PyTorch两种深度学习框架
- **Web界面**：直观的用户界面，支持图片上传和实时训练监控
- **灵活的参数调节**：可自定义内容权重、风格权重和训练轮数
- **实时进度跟踪**：可视化训练进度和损失曲线
- **快速风格迁移**：包含预训练模型的快速风格迁移实现
- **多种风格模型**：提供candy、mosaic、rain_princess、udnie等多种艺术风格

## 技术栈

- **后端**：Flask + PyTorch/Keras
- **前端**：HTML/CSS/JavaScript
- **深度学习框架**：PyTorch, TensorFlow/Keras
- **模型架构**：VGG19
- **损失函数**：内容损失、风格损失、总变差损失

## 项目结构

```
style_transfer/
├── app.py                     # Flask Web应用主入口
├── style_transfer_by_torch.py # PyTorch风格迁移实现
├── style_transfer_by_keras.py # Keras风格迁移实现
├── input.html                 # 参数设置页面
├── training.html              # 训练过程页面
├── fast_style/               # 快速风格迁移模块
│   ├── single_fast_style.py   # 单文件快速风格迁移
│   ├── pyproject.toml         # 项目依赖配置
│   └── saved_models/          # 预训练模型
├── images/                   # 示例图片
├── results/                  # 输出结果
└── README.md                 # 项目说明
```

## 功能模块

### 1. Web界面
- **输入页面**：支持内容图片和风格图片上传
- **参数调节**：内容权重、风格权重、训练轮数
- **训练监控**：实时显示训练进度、损失曲线和中间结果
- **结果展示**：最终风格迁移结果展示

### 2. 风格迁移算法
#### 传统优化方法（PyTorch/Keras）
- 使用预训练的VGG19网络提取特征
- 内容损失：衡量内容图像和合成图像在高层特征上的差异
- 风格损失：通过Gram矩阵衡量风格图像和合成图像在纹理上的差异
- 总变差损失：使生成图像更加平滑自然
- 优化算法：L-BFGS或Adam优化器

#### 快速前馈方法
- 基于Transformer网络的实时风格迁移
- 支持多种预训练风格模型（candy、mosaic、rain_princess、udnie等）
- 一次性前馈，无需迭代优化
- 推理速度快，适合实时应用场景

### 3. 核心组件

#### 内容损失计算
```python
class ContentLoss(nn.Module):
    def __init__(self, content_feature, weight):
        super(ContentLoss, self).__init__()
        self.content_feature = content_feature.detach()
        self.criterion = nn.MSELoss()
        self.weight = weight
    
    def forward(self, combination):
        self.loss = self.criterion(
            combination.clone() * self.weight,
            self.content_feature.clone() * self.weight
        )
        return combination
```

#### 风格损失计算
```python
class StyleLoss(nn.Module):
    def __init__(self, style_feature, weight):
        super(StyleLoss, self).__init__()
        self.style_feature = style_feature.detach()
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, combination):
        style_feature = self.gram(self.style_feature.clone() * self.weight)
        combination_features = self.gram(combination.clone() * self.weight)
        self.loss = self.criterion(combination_features, style_feature)
        return combination
```

#### Gram矩阵计算
```python
class GramMatrix(nn.Module):
    def forward(self, input):
        b, n, h, w = input.size()  
        features = input.view(b * n, h * w) 
        G = torch.mm(features, features.t()) 
        return G.div(b * n * h * w)
```

## 使用方法

### 1. 环境准备

#### 安装依赖
```bash
pip install flask flask-cors pillow torch torchvision tensorflow keras
```

#### 准备预训练模型
快速风格迁移功能需要预训练模型才能运行，具体下载方式见 [快速风格迁移模型详解](#预训练模型下载) 部分。

### 2. 运行Web服务
```bash
python app.py
```
访问 `http://localhost:5000` 打开Web界面

### 3. 使用Web界面
1. 在输入页面上传内容图片和风格图片
2. 调整参数（内容权重、风格权重、训练轮数）
3. 点击"开始训练"启动风格迁移过程
4. 在训练页面实时查看进度和中间结果

### 4. 命令行快速风格迁移

#### 基础使用
```bash
# 使用所有可用风格
python fast_style/single_fast_style.py --content images/Cat.jpg

# 指定特定风格
python fast_style/single_fast_style.py --content images/Cat.jpg --style candy

# 自定义参数
python fast_style/single_fast_style.py --content images/Cat.jpg --style mosaic \
    --weights fast_style/saved_models --output-dir fast_style/fast_results --max-size 960
```

#### 参数说明
- `--content`: 输入图片路径（默认: images/Cat.jpg）
- `--style`: 使用的风格（可选: candy/mosaic/rain_princess/udnie/all）（默认: all）
- `--weights`: 预训练权重目录（默认: fast_style/saved_models）
- `--output-dir`: 输出目录（默认: fast_style/fast_results）
- `--max-size`: 最长边缩放阈值，控制推理开销（默认: 720）

#### 运行示例
```bash
# 使用Taipei101.jpg图片，应用所有风格
python fast_style/single_fast_style.py --content images/Taipei101.jpg

# 仅应用candy风格，输出到results目录
python fast_style/single_fast_style.py --content images/Taipei101.jpg --style candy --output-dir results

# 自定义最大尺寸为1024像素
python fast_style/single_fast_style.py --content images/Cat.jpg --max-size 1024
```

## 算法原理

### 风格迁移基本概念
图像风格迁移的目标是将一幅内容图像的内容与另一幅风格图像的风格相结合，生成一幅新的图像，既保留内容图像的主要内容，又具有风格图像的艺术风格。

### 损失函数组成
1. **内容损失**：确保合成图像保留内容图像的主要内容
2. **风格损失**：确保合成图像具有风格图像的纹理和颜色特征
3. **总变差损失**：提高图像的平滑性，减少噪声

### VGG19网络应用
- 使用预训练的VGG19网络提取图像特征
- 内容损失通常在较深层（如block4_conv2）计算
- 风格损失在多个层（block1_conv1到block5_conv1）上计算

## 模型架构

### VGG19特征提取
- 基于VGG19的卷积层提取多尺度特征
- 不同层捕获不同级别的纹理和内容信息
- 风格表示通过Gram矩阵捕获特征间的相关性

### 快速风格迁移网络
- 编码器：多层卷积提取特征
- 残差块：保持特征信息
- 解码器：上采样重建图像
- 归一化：实例归一化提升效果

## 预训练模型
- 项目使用预训练的VGG19模型作为特征提取器
- 快速风格迁移使用预训练的Transformer模型
- 模型文件需要单独下载或训练获得

## 快速风格迁移模型详解

### 网络架构
快速风格迁移网络采用编码器-变换器-解码器架构：

1. **编码器**：通过多层卷积层提取图像特征
   - ConvLayer(3, 32, kernel_size=9, stride=1)
   - ConvLayer(32, 64, kernel_size=3, stride=2)
   - ConvLayer(64, 128, kernel_size=3, stride=2)

2. **变换器**：包含5个残差块进行风格变换
   - 每个残差块包含两个卷积层和实例归一化层
   - 使用ReLU激活函数

3. **解码器**：通过反卷积层重建图像
   - UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
   - UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
   - ConvLayer(32, 3, kernel_size=9, stride=1)

### 支持的风格模型

| 风格名称 | 描述 | 文件名 |
|---------|------|--------|
| Candy | 高饱和度、强涂抹感的风格 | candy.pth |
| Mosaic | 马赛克拼贴，色块层次分明 | mosaic.pth |
| Rain Princess | 油画式流动笔触 | rain_princess.pth |
| Udnie | 抽象色块风格 | udnie.pth |

### 预训练模型下载

您可以从以下链接下载预训练的快速风格迁移模型：

- **Candy模型**: [Download](https://drive.google.com/uc?id=0B9CKOTmy0DyaNnBlTH9JUnRwVFk)
- **Mosaic模型**: [Download](https://drive.google.com/uc?id=0B9CKOTmy0DyaQ0wwSVc4MlE5UEk)
- **Rain Princess模型**: [Download](https://drive.google.com/uc?id=0B9CKOTmy0DyaYlEyM0JuOGQtUlE)
- **Udnie模型**: [Download](https://drive.google.com/uc?id=0B9CKOTmy0DyaWlJHeHhOQXNYaG8)

下载后请将模型文件放置在 `fast_style/saved_models/` 目录下。

## 应用场景
- 艺术创作辅助
- 图像编辑和美化
- 风格化滤镜应用
- 创意设计工具

## 性能优化
- 支持GPU加速训练和推理
- 可调节图像尺寸以平衡质量和速度
- 提供快速前馈模型实现实时风格迁移

## 快速风格迁移 vs 传统方法

| 特性 | 传统优化方法 | 快速前馈方法 |
|------|------------|------------|
| 推理速度 | 较慢（需要数百次迭代） | 快速（单次前馈） |
| GPU需求 | 高（长时间占用） | 低（短时间占用） |
| 内存占用 | 中等 | 低 |
| 质量 | 高 | 高 |
| 适用场景 | 高质量要求、离线处理 | 实时应用、批量处理 |

快速风格迁移方法特别适用于需要实时处理或批量处理的场景，而传统优化方法更适合对单张图片追求最高质量的场景。

## 注意事项
- 训练过程可能消耗大量计算资源
- 建议使用GPU加速以获得更好的性能
- 预训练模型文件较大，需要预先下载
- 不同类型的图片可能需要调整参数以获得最佳效果

## 未来改进方向
- 支持更多风格模型
- 优化模型架构以提高效率
- 增加更多的损失函数选项
- 提供移动端部署方案

---

该项目展示了深度学习在计算机视觉领域的强大能力，通过神经网络实现了令人惊叹的图像风格迁移效果。