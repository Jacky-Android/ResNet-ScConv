# ResNet-ScConv
This is the unofficial implementation of SCConv: Spatial and Channel Reconstruction Convolution for Feature Redundancy .

Update: Solved some bugs in SRU, thanks to @huihui43 , zhangzh16.
Thanks to:<https://github.com/cheng-haha/ScConv/tree/main> 
## paper<http://openaccess.thecvf.com//content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf>

![image](https://github.com/Jacky-Android/ResNet-ScConv/assets/55181594/9386ee15-6e84-4279-a833-18729481f6ee)

## Actually, you just need to insert the ScConv into your model anywhere. So we add to ResNet


====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
ResNet                                             [1, 4]                    --
├─Conv2d: 1-1                                      [1, 64, 256, 256]         9,408
├─BatchNorm2d: 1-2                                 [1, 64, 256, 256]         128
├─ReLU: 1-3                                        [1, 64, 256, 256]         --
├─MaxPool2d: 1-4                                   [1, 64, 128, 128]         --
├─Sequential: 1-5                                  [1, 64, 128, 128]         --
│    └─BasicBlock: 2-1                             [1, 64, 128, 128]         --
│    │    └─Conv2d: 3-1                            [1, 64, 128, 128]         36,864
│    │    └─BatchNorm2d: 3-2                       [1, 64, 128, 128]         128
│    │    └─ReLU: 3-3                              [1, 64, 128, 128]         --
│    │    └─Conv2d: 3-4                            [1, 64, 128, 128]         36,864
│    │    └─ScConv: 3-5                            [1, 64, 128, 128]         7,616
│    │    └─BatchNorm2d: 3-6                       [1, 64, 128, 128]         128
│    │    └─ReLU: 3-7                              [1, 64, 128, 128]         --
│    └─BasicBlock: 2-2                             [1, 64, 128, 128]         --
│    │    └─Conv2d: 3-8                            [1, 64, 128, 128]         36,864
│    │    └─BatchNorm2d: 3-9                       [1, 64, 128, 128]         128
│    │    └─ReLU: 3-10                             [1, 64, 128, 128]         --
│    │    └─Conv2d: 3-11                           [1, 64, 128, 128]         36,864
│    │    └─ScConv: 3-12                           [1, 64, 128, 128]         7,616
│    │    └─BatchNorm2d: 3-13                      [1, 64, 128, 128]         128
│    │    └─ReLU: 3-14                             [1, 64, 128, 128]         --
│    └─BasicBlock: 2-3                             [1, 64, 128, 128]         --
│    │    └─Conv2d: 3-15                           [1, 64, 128, 128]         36,864
│    │    └─BatchNorm2d: 3-16                      [1, 64, 128, 128]         128
│    │    └─ReLU: 3-17                             [1, 64, 128, 128]         --
│    │    └─Conv2d: 3-18                           [1, 64, 128, 128]         36,864
│    │    └─ScConv: 3-19                           [1, 64, 128, 128]         7,616
│    │    └─BatchNorm2d: 3-20                      [1, 64, 128, 128]         128
│    │    └─ReLU: 3-21                             [1, 64, 128, 128]         --
├─Sequential: 1-6                                  [1, 128, 64, 64]          --
│    └─BasicBlock: 2-4                             [1, 128, 64, 64]          --
│    │    └─Sequential: 3-22                       [1, 128, 64, 64]          8,448
│    │    └─Conv2d: 3-23                           [1, 128, 64, 64]          73,728
│    │    └─BatchNorm2d: 3-24                      [1, 128, 64, 64]          256
│    │    └─ReLU: 3-25                             [1, 128, 64, 64]          --
│    │    └─Conv2d: 3-26                           [1, 128, 64, 64]          147,456
│    │    └─ScConv: 3-27                           [1, 128, 64, 64]          30,080
│    │    └─BatchNorm2d: 3-28                      [1, 128, 64, 64]          256
│    │    └─ReLU: 3-29                             [1, 128, 64, 64]          --
│    └─BasicBlock: 2-5                             [1, 128, 64, 64]          --
│    │    └─Conv2d: 3-30                           [1, 128, 64, 64]          147,456
│    │    └─BatchNorm2d: 3-31                      [1, 128, 64, 64]          256
│    │    └─ReLU: 3-32                             [1, 128, 64, 64]          --
│    │    └─Conv2d: 3-33                           [1, 128, 64, 64]          147,456
│    │    └─ScConv: 3-34                           [1, 128, 64, 64]          30,080
│    │    └─BatchNorm2d: 3-35                      [1, 128, 64, 64]          256
│    │    └─ReLU: 3-36                             [1, 128, 64, 64]          --
│    └─BasicBlock: 2-6                             [1, 128, 64, 64]          --
│    │    └─Conv2d: 3-37                           [1, 128, 64, 64]          147,456
│    │    └─BatchNorm2d: 3-38                      [1, 128, 64, 64]          256
│    │    └─ReLU: 3-39                             [1, 128, 64, 64]          --
│    │    └─Conv2d: 3-40                           [1, 128, 64, 64]          147,456
│    │    └─ScConv: 3-41                           [1, 128, 64, 64]          30,080
│    │    └─BatchNorm2d: 3-42                      [1, 128, 64, 64]          256
│    │    └─ReLU: 3-43                             [1, 128, 64, 64]          --
│    └─BasicBlock: 2-7                             [1, 128, 64, 64]          --
│    │    └─Conv2d: 3-44                           [1, 128, 64, 64]          147,456
│    │    └─BatchNorm2d: 3-45                      [1, 128, 64, 64]          256
│    │    └─ReLU: 3-46                             [1, 128, 64, 64]          --
│    │    └─Conv2d: 3-47                           [1, 128, 64, 64]          147,456
│    │    └─ScConv: 3-48                           [1, 128, 64, 64]          30,080
│    │    └─BatchNorm2d: 3-49                      [1, 128, 64, 64]          256
│    │    └─ReLU: 3-50                             [1, 128, 64, 64]          --
├─Sequential: 1-7                                  [1, 256, 32, 32]          --
│    └─BasicBlock: 2-8                             [1, 256, 32, 32]          --
│    │    └─Sequential: 3-51                       [1, 256, 32, 32]          33,280
│    │    └─Conv2d: 3-52                           [1, 256, 32, 32]          294,912
│    │    └─BatchNorm2d: 3-53                      [1, 256, 32, 32]          512
│    │    └─ReLU: 3-54                             [1, 256, 32, 32]          --
│    │    └─Conv2d: 3-55                           [1, 256, 32, 32]          589,824
│    │    └─ScConv: 3-56                           [1, 256, 32, 32]          119,552
│    │    └─BatchNorm2d: 3-57                      [1, 256, 32, 32]          512
│    │    └─ReLU: 3-58                             [1, 256, 32, 32]          --
│    └─BasicBlock: 2-9                             [1, 256, 32, 32]          --
│    │    └─Conv2d: 3-59                           [1, 256, 32, 32]          589,824
│    │    └─BatchNorm2d: 3-60                      [1, 256, 32, 32]          512
│    │    └─ReLU: 3-61                             [1, 256, 32, 32]          --
│    │    └─Conv2d: 3-62                           [1, 256, 32, 32]          589,824
│    │    └─ScConv: 3-63                           [1, 256, 32, 32]          119,552
│    │    └─BatchNorm2d: 3-64                      [1, 256, 32, 32]          512
│    │    └─ReLU: 3-65                             [1, 256, 32, 32]          --
│    └─BasicBlock: 2-10                            [1, 256, 32, 32]          --
│    │    └─Conv2d: 3-66                           [1, 256, 32, 32]          589,824
│    │    └─BatchNorm2d: 3-67                      [1, 256, 32, 32]          512
│    │    └─ReLU: 3-68                             [1, 256, 32, 32]          --
│    │    └─Conv2d: 3-69                           [1, 256, 32, 32]          589,824
│    │    └─ScConv: 3-70                           [1, 256, 32, 32]          119,552
│    │    └─BatchNorm2d: 3-71                      [1, 256, 32, 32]          512
│    │    └─ReLU: 3-72                             [1, 256, 32, 32]          --
│    └─BasicBlock: 2-11                            [1, 256, 32, 32]          --
│    │    └─Conv2d: 3-73                           [1, 256, 32, 32]          589,824
│    │    └─BatchNorm2d: 3-74                      [1, 256, 32, 32]          512
│    │    └─ReLU: 3-75                             [1, 256, 32, 32]          --
│    │    └─Conv2d: 3-76                           [1, 256, 32, 32]          589,824
│    │    └─ScConv: 3-77                           [1, 256, 32, 32]          119,552
│    │    └─BatchNorm2d: 3-78                      [1, 256, 32, 32]          512
│    │    └─ReLU: 3-79                             [1, 256, 32, 32]          --
│    └─BasicBlock: 2-12                            [1, 256, 32, 32]          --
│    │    └─Conv2d: 3-80                           [1, 256, 32, 32]          589,824
│    │    └─BatchNorm2d: 3-81                      [1, 256, 32, 32]          512
│    │    └─ReLU: 3-82                             [1, 256, 32, 32]          --
│    │    └─Conv2d: 3-83                           [1, 256, 32, 32]          589,824
│    │    └─ScConv: 3-84                           [1, 256, 32, 32]          119,552
│    │    └─BatchNorm2d: 3-85                      [1, 256, 32, 32]          512
│    │    └─ReLU: 3-86                             [1, 256, 32, 32]          --
│    └─BasicBlock: 2-13                            [1, 256, 32, 32]          --
│    │    └─Conv2d: 3-87                           [1, 256, 32, 32]          589,824
│    │    └─BatchNorm2d: 3-88                      [1, 256, 32, 32]          512
│    │    └─ReLU: 3-89                             [1, 256, 32, 32]          --
│    │    └─Conv2d: 3-90                           [1, 256, 32, 32]          589,824
│    │    └─ScConv: 3-91                           [1, 256, 32, 32]          119,552
│    │    └─BatchNorm2d: 3-92                      [1, 256, 32, 32]          512
│    │    └─ReLU: 3-93                             [1, 256, 32, 32]          --
├─Sequential: 1-8                                  [1, 512, 16, 16]          --
│    └─BasicBlock: 2-14                            [1, 512, 16, 16]          --
│    │    └─Sequential: 3-94                       [1, 512, 16, 16]          132,096
│    │    └─Conv2d: 3-95                           [1, 512, 16, 16]          1,179,648
│    │    └─BatchNorm2d: 3-96                      [1, 512, 16, 16]          1,024
│    │    └─ReLU: 3-97                             [1, 512, 16, 16]          --
│    │    └─Conv2d: 3-98                           [1, 512, 16, 16]          2,359,296
│    │    └─ScConv: 3-99                           [1, 512, 16, 16]          476,672
│    │    └─BatchNorm2d: 3-100                     [1, 512, 16, 16]          1,024
│    │    └─ReLU: 3-101                            [1, 512, 16, 16]          --
│    └─BasicBlock: 2-15                            [1, 512, 16, 16]          --
│    │    └─Conv2d: 3-102                          [1, 512, 16, 16]          2,359,296
│    │    └─BatchNorm2d: 3-103                     [1, 512, 16, 16]          1,024
│    │    └─ReLU: 3-104                            [1, 512, 16, 16]          --
│    │    └─Conv2d: 3-105                          [1, 512, 16, 16]          2,359,296
│    │    └─ScConv: 3-106                          [1, 512, 16, 16]          476,672
│    │    └─BatchNorm2d: 3-107                     [1, 512, 16, 16]          1,024
│    │    └─ReLU: 3-108                            [1, 512, 16, 16]          --
│    └─BasicBlock: 2-16                            [1, 512, 16, 16]          --
│    │    └─Conv2d: 3-109                          [1, 512, 16, 16]          2,359,296
│    │    └─BatchNorm2d: 3-110                     [1, 512, 16, 16]          1,024
│    │    └─ReLU: 3-111                            [1, 512, 16, 16]          --
│    │    └─Conv2d: 3-112                          [1, 512, 16, 16]          2,359,296
│    │    └─ScConv: 3-113                          [1, 512, 16, 16]          476,672
│    │    └─BatchNorm2d: 3-114                     [1, 512, 16, 16]          1,024
│    │    └─ReLU: 3-115                            [1, 512, 16, 16]          --
├─AdaptiveAvgPool2d: 1-9                           [1, 512, 1, 1]            --
├─Linear: 1-10                                     [1, 4]                    2,052
====================================================================================================
Total params: 23,577,220
Trainable params: 23,577,220
Non-trainable params: 0
Total mult-adds (G): 21.09
====================================================================================================
Input size (MB): 3.15
Forward/backward pass size (MB): 557.58
Params size (MB): 94.31
Estimated Total Size (MB): 655.03
====================================================================================================
