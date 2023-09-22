from torchinfo import summary

#from ori import UNetFormer
from resnet import resnet101,resnet34,resnet50,resnext101_32x8d,resnext50_32x4d
model = resnet34(num_classes=4)

summary(model=model,input_size=(1,3,512,512))