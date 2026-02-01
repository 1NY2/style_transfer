# app.py
import os
import io
import base64
import json
import threading

import torch
import torch.nn as nn
from flask import Flask, render_template, request, Response, send_from_directory
from flask_cors import CORS
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time

app = Flask(__name__)
CORS(app)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContentLoss(nn.Module):
    def __init__(self, content_feature, weight):
        super(ContentLoss, self).__init__()
        self.content_feature = content_feature.detach()
        self.criterion = nn.MSELoss()
        self.weight = weight

    def forward(self, combination):
        self.loss = self.criterion(combination.clone() * self.weight,
                                   self.content_feature.clone() * self.weight)
        return combination


class GramMatrix(nn.Module):
    def forward(self, input):
        b, n, h, w = input.size()
        features = input.view(b * n, h * w)
        G = torch.mm(features, features.t())
        return G.div(b * n * h * w)


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


class StyleTransfer:
    def __init__(self, content_weight=0.025, style_weight=5):
        self.content_weight = content_weight
        self.style_weight = style_weight

        # 加载VGG19模型 (使用本地权重)
        self.vgg19 = models.vgg19()
        # 检查本地权重文件是否存在
        if os.path.exists('vgg19-dcbb9e9d.pth'):
            print("正在加载本地VGG19模型权重...")
            self.vgg19.load_state_dict(torch.load('vgg19-dcbb9e9d.pth'))
        else:
            # 如果本地文件不存在，则下载预训练权重
            print("本地权重文件不存在，正在下载预训练权重...")
            self.vgg19 = models.vgg19(pretrained=True)

        self.img_ncols = 400
        self.img_nrows = 300
        print("VGG19模型加载完成")

    def process_img(self, img_data):
        # 解码base64图像
        if img_data.startswith('data:image'):
            img_data = img_data.split(',')[1]

        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        loader = transforms.Compose([
            transforms.Resize((self.img_nrows, self.img_ncols)),
            transforms.ToTensor()
        ])

        img_tensor = loader(img)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.to(device, torch.float)

    def deprocess_img(self, tensor):
        """将张量转换为base64图像"""
        unloader = transforms.ToPILImage()
        tensor = tensor.cpu().clone()
        img_tensor = tensor.squeeze(0)
        img = unloader(img_tensor)

        # 转换为base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def get_loss_and_model(self, vgg_model, content_image, style_image):
        vgg_layers = vgg_model.features.to(device).eval()
        style_losses = []
        content_losses = []
        model = nn.Sequential()

        style_layer_name_mapping = {
            '0': "style_loss_1",    # conv1_1 → 捕捉最基础纹理
            '5': "style_loss_2",    # conv2_1 → 中等纹理
            '10': "style_loss_3",   # conv3_1 → 复杂纹理
            '19': "style_loss_4",   # conv4_1 → 高级风格
            '28': "style_loss_5",   # conv5_1 → 最细风格
        }

        content_layer_name_mapping = {'30': "content_loss"}

        for name, module in vgg_layers._modules.items():
            model.add_module(name, module)

            if name in content_layer_name_mapping:
                content_feature = model(content_image).clone()
                content_loss = ContentLoss(content_feature, self.content_weight)
                model.add_module(content_layer_name_mapping[name], content_loss)
                content_losses.append(content_loss)

            if name in style_layer_name_mapping:
                style_feature = model(style_image).clone()
                style_loss = StyleLoss(style_feature, self.style_weight)
                style_losses.append(style_loss)
                model.add_module(style_layer_name_mapping[name], style_loss)

        return content_losses, style_losses, model

    def get_input_param_optimizer(self, input_img):
        input_param = nn.Parameter(input_img.data)
        optimizer = torch.optim.LBFGS([input_param])
        return input_param, optimizer

    def train(self, content_image_data, style_image_data, epochs, callback=None):
        print(f"开始训练，内容图片: {content_image_data[:50]}..., 风格图片: {style_image_data[:50]}..., 轮数: {epochs}")

        # 处理图像
        content_tensor = self.process_img(content_image_data)
        style_tensor = self.process_img(style_image_data)
        combination_tensor = content_tensor.clone()

        print(f"图像处理完成，内容张量形状: {content_tensor.shape}, 风格张量形状: {style_tensor.shape}")

        # 获取损失函数和模型
        content_losses, style_losses, model = self.get_loss_and_model(
            self.vgg19, content_tensor, style_tensor)

        print(f"模型构建完成，内容损失数量: {len(content_losses)}, 风格损失数量: {len(style_losses)}")

        # 获取输入参数和优化器
        combination_param, optimizer = self.get_input_param_optimizer(combination_tensor)

        cur, pre = 10, 10

        for i in range(1, epochs + 1):
            start = time.time()
            print(f"开始第 {i} 轮训练")

            def closure():
                combination_param.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(combination_param)

                style_score = 0
                content_score = 0

                for cl in content_losses:
                    content_score += cl.loss
                for sl in style_losses:
                    style_score += sl.loss

                loss = content_score + style_score
                loss.backward()
                return loss.detach()

            loss = optimizer.step(closure)
            cur, pre = loss.item(), cur
            end = time.time()

            print(f"第 {i} 轮训练完成，损失: {loss.item()}, 耗时: {end - start:.2f}s")

            # 每10个epoch发送一次中间结果
            if i % 10 == 0 or i == 1:
                result_img = self.deprocess_img(combination_param)
                if callback:
                    callback({
                        'type': 'result',
                        'epoch': i,
                        'loss': loss.item(),
                        'image': result_img
                    })

            # 发送进度更新
            if callback:
                callback({
                    'type': 'progress',
                    'epoch': i,
                    'loss': loss.item(),
                    'time': int(end - start)
                })

            # 早停机制
            if pre <= cur:
                if callback:
                    callback({
                        'type': 'message',
                        'text': 'Early stopping!'
                    })
                break

            combination_param.data.clamp_(0, 1)

        # 发送最终结果
        final_img = self.deprocess_img(combination_param)
        if callback:
            callback({
                'type': 'complete',
                'epoch': epochs,
                'loss': cur,
                'image': final_img
            })

@app.route('/')
def index():
    return send_from_directory('.', 'input.html')


@app.route('/input.html')
def input_page():
    return send_from_directory('.', 'input.html')

@app.route('/training.html')
def training_page():
    return send_from_directory('.', 'training.html')

@app.route('/train', methods=['POST'])
def train_style_transfer():
    try:
        data = request.json

        if not data:
            return Response(json.dumps({
                'type': 'error',
                'message': '未接收到训练数据'
            }) + '\n', mimetype='text/plain')

        # 验证图片数据是否完整
        if not data['contentImage'] or not data['styleImage']:
            return Response(json.dumps({
                'type': 'error',
                'message': '图片数据不完整'
            }) + '\n', mimetype='text/plain')

        # 验证图片数据是否有效
        try:
            # 尝试解码base64数据
            if data['contentImage'].startswith('data:image'):
                img_data = data['contentImage'].split(',')[1]
            else:
                img_data = data['contentImage']

            base64.b64decode(img_data)
        except Exception as e:
            return Response(json.dumps({
                'type': 'error',
                'message': '图片数据无效: ' + str(e)
            }) + '\n', mimetype='text/plain')

        # 检查必需的数据
        required_fields = ['contentImage', 'styleImage']
        for field in required_fields:
            if field not in data:
                return Response(json.dumps({
                    'type': 'error',
                    'message': f'缺少必要参数: {field}'
                }) + '\n', mimetype='text/plain')

        content_weight = float(data.get('contentWeight', 0.025))
        style_weight = float(data.get('styleWeight', 5))
        epochs = int(data.get('epochs', 100))

        # 初始化风格迁移模型
        st = StyleTransfer(content_weight, style_weight)

        # 创建一个队列用于传递结果
        import queue
        result_queue = queue.Queue()

        # 结束标志
        END_MARKER = object()

        def send_result(output):
            result_queue.put(output)

        def generate():
            # 启动训练线程
            import threading
            def train_thread():
                try:
                    st.train(
                        data['contentImage'],
                        data['styleImage'],
                        epochs,
                        send_result
                    )
                except Exception as e:
                    send_result({
                        'type': 'error',
                        'message': f'训练过程中发生错误: {str(e)}'
                    })
                finally:
                    result_queue.put(END_MARKER)

            thread = threading.Thread(target=train_thread)
            thread.start()

            # 实时获取并发送结果
            while True:
                try:
                    result = result_queue.get(timeout=1)
                    if result is END_MARKER:
                        break
                    # 确保JSON字符串完整且正确编码
                    json_str = json.dumps(result, ensure_ascii=False)
                    yield f"{json_str}\n".encode('utf-8')
                except queue.Empty:
                    # 检查线程是否还在运行
                    if not thread.is_alive():
                        break
                    continue

            # 等待线程结束
            thread.join()

        return Response(generate(), mimetype='text/plain', direct_passthrough=True)

    except Exception as e:
        return Response(json.dumps({
            'type': 'error',
            'message': f'请求处理错误: {str(e)}'
        }) + '\n', mimetype='text/plain')


@app.route('/test')
def test():
    return json.dumps({'status': 'ok', 'message': 'Server is running'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
