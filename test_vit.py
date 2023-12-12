import torch
import torch.nn as nn
from vit_pytorch import ViT

class CustomViT(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super(CustomViT, self).__init__()
        self.pretrained_vit = ViT(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=in_channels
        )

    def forward(self, x):
        print("Original shape:", x.shape)  # 打印原始形状 2,512,256,16

        # 划分 patches
        x_patches = x.unfold(2, 16, 16).unfold(3, 16, 16)
        batch_size, channels, h_patches, w_patches, patch_h, patch_w = x_patches.shape
        print("After unfolding:", x_patches.shape)  # 打印展开后的形状 2, 512, 16, 1, 16, 16

        x_patches = x_patches.contiguous().view(batch_size, channels, h_patches * w_patches, patch_h * patch_w)
        print("After view:", x_patches.shape)  # 打印变形后的形状 2, 512, 16, 256

        x_patches = x_patches.permute(0, 2, 1, 3)
        print("After permute:", x_patches.shape)  # 打印排列后的形状 2, 16, 512, 256

        x_patches = x_patches.flatten(2)
        print("After flatten:", x_patches.shape)  # 打印扁平化后的形状 2, 16, 131072

        vit_output = self.pretrained_vit(x_patches)
        print("After ViT:", vit_output.shape)  # 打印通过 ViT 后的形状

        output = vit_output.view(batch_size, channels, 256, 16)
        print("Final output shape:", output.shape)  # 打印最终输出的形状

        return output

# 初始化模型
model = CustomViT(
    in_channels=512,
    img_size=16,
    patch_size=4,
    num_classes=1,
    dim=1024,
    depth=6,
    heads=8,
    mlp_dim=2048
)

# 假设输入数据
input_data = torch.randn(2, 512, 256, 16)

# 通过模型
output_data = model(input_data)
