"""
单文件快速风格迁移脚本

基础运行（默认一次输出所有风格）
    python single_fast_style.py

指定参数示例
    python single_fast_style.py --content images/Feathers.jpg --style candy \
        --weights saved_models --output-dir fast_results --max-size 960

参数说明
    --content     输入图片路径，默认 images/Taipei101.jpg
    --style       使用的风格：candy/mosaic/rain_princess/udnie/all（默认 all）
    --weights     预训练权重目录（默认 saved_models）
    --output-dir  输出目录，文件命名为 原图名_风格.png（默认 fast_results）
    --max-size    最长边缩放阈值，控制推理开销，默认 720

必备文件
    - 4 个预训练权重（默认放在 saved_models/ 下）：
        candy.pth, mosaic.pth, rain_princess.pth, udnie.pth
    - 输入图片（JPEG/PNG 均可）
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torchvision import transforms


# -----------------------------
# 网络结构定义
# -----------------------------
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.reflection_pad(x)
        return self.conv2d(x)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, upsample: int | None = None) -> None:
        super().__init__()
        self.upsample = upsample
        padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.upsample:
            x = torch.nn.functional.interpolate(x, mode="nearest", scale_factor=self.upsample)
        x = self.reflection_pad(x)
        return self.conv2d(x)


class TransformerNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        return self.deconv3(y)


# -----------------------------
# 推理逻辑
# -----------------------------
FAST_MODELS: Dict[str, Dict[str, str]] = {
    "candy": {
        "name": "Candy",
        "filename": "candy.pth",
        "description": "高饱和度、强涂抹感的风格",
    },
    "mosaic": {
        "name": "Mosaic",
        "filename": "mosaic.pth",
        "description": "马赛克拼贴，色块层次分明",
    },
    "rain_princess": {
        "name": "Rain Princess",
        "filename": "rain_princess.pth",
        "description": "油画式流动笔触",
    },
    "udnie": {
        "name": "Udnie",
        "filename": "udnie.pth",
        "description": "抽象色块风格",
    },
}


def load_transformer(model_id: str, weights_dir: Path, device: torch.device) -> TransformerNet:
    if model_id not in FAST_MODELS:
        raise ValueError(f"未知模型: {model_id}. 可选: {', '.join(FAST_MODELS)}")

    model_path = weights_dir / FAST_MODELS[model_id]["filename"]
    if not model_path.exists():
        raise FileNotFoundError(f"未找到权重文件: {model_path}")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using `torch.load` with `weights_only=False`",
            category=FutureWarning,
        )
        state_dict = torch.load(model_path, map_location=device)
    # 清理与 InstanceNorm 相关的运行状态键
    for key in list(state_dict.keys()):
        if "running_mean" in key or "running_var" in key:
            if "in" in key:
                del state_dict[key]

    model = TransformerNet()
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def prepare_tensor(image_path: Path, device: torch.device, max_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    longest = max(image.size)
    if longest > max_size:
        scale = max_size / longest
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.BICUBIC)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ])
    tensor = preprocess(image).unsqueeze(0).to(device)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0, 255).permute(1, 2, 0).numpy().astype("uint8")
    return Image.fromarray(array)


def stylize_all(content_image: Path, output_dir: Path, styles: list[str], weights_dir: Path, max_size: int = 720) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)
    content_tensor = prepare_tensor(content_image, device, max_size)
    content_name = content_image.stem

    summary = []
    for style_id in styles:
        print("-" * 60)
        print(f"应用风格: {style_id} ({FAST_MODELS[style_id]['name']})")
        model = load_transformer(style_id, weights_dir, device)
        with torch.no_grad():
            start = time.time()
            output = model(content_tensor).cpu()[0]
            duration = time.time() - start

        filename = f"{content_name}_{style_id}.png"
        output_path = output_dir / filename
        tensor_to_image(output).save(output_path)
        print(f"✓ 保存到 {output_path} | 耗时 {duration:.3f}s")
        summary.append((FAST_MODELS[style_id]['name'], output_path, duration))

    print("=" * 60)
    print("处理完成，总结：")
    total = 0.0
    for name, path, cost in summary:
        total += cost
        print(f"• {name:<15} -> {path} ({cost:.2f}s)")
    avg = total / len(summary) if summary else 0
    print(f"总耗时: {total:.2f}s | 平均耗时: {avg:.2f}s/张")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单文件快速风格迁移推理脚本")
    parser.add_argument(
        "--content",
        default="images/Cat.jpg",
        help="输入图片路径，可在代码或参数中修改",
    )
    style_choices = list(FAST_MODELS.keys()) + ["all"]
    parser.add_argument(
        "--style",
        default="all",
        choices=style_choices,
        help="使用的风格模型，默认一次输出全部",
    )
    parser.add_argument("--weights", default="fast_style/saved_models", help="预训练权重所在目录")
    parser.add_argument(
        "--output-dir",
        default="fast_style/fast_results",
        help="输出目录，结果文件自动命名为 原图名_风格.png",
    )
    parser.add_argument("--max-size", type=int, default=720, help="最长边缩放到该尺寸内")
    return parser.parse_args()


def main() -> None:
    import os
    from pathlib import Path
    
    args = parse_args()
    
    # 处理相对路径，确保它们相对于项目根目录
    project_root = Path(__file__).parent.parent  # 获取项目根目录
    
    # 如果是相对路径（不以 / 开头），则相对于项目根目录
    if str(args.content).startswith('fast_style/'):
        content_path = project_root / args.content
    elif not Path(args.content).is_absolute():
        content_path = project_root / args.content
    else:
        content_path = Path(args.content)
        
    if not content_path.exists():
        raise FileNotFoundError(f"输入图片不存在: {content_path}")

    if str(args.weights).startswith('fast_style/'):
        weights_dir = project_root / args.weights
    elif not Path(args.weights).is_absolute():
        weights_dir = project_root / args.weights
    else:
        weights_dir = Path(args.weights)
        
    if not weights_dir.exists():
        raise FileNotFoundError(f"权重目录不存在: {weights_dir}")

    if str(args.output_dir).startswith('fast_style/'):
        output_dir = project_root / args.output_dir
    elif not Path(args.output_dir).is_absolute():
        output_dir = project_root / args.output_dir
    else:
        output_dir = Path(args.output_dir)

    print("=" * 60)
    print(f"输入图片: {content_path}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    if args.style == "all":
        styles = list(FAST_MODELS.keys())
    else:
        styles = [args.style]

    stylize_all(content_path, output_dir, styles, weights_dir, args.max_size)


if __name__ == "__main__":
    main()
