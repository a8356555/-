#!/bin/bash
for MODEL in resnet101 resnet50 efficientnet-b4 efficientnet-b5 efficientnet-b6 efficientnet-b7
do
#    cd /content/YuShanAICompetition/csrc/serialize_engine_from_onnx/bin
#    ./trt_serialize /content/$MODEL.onnx /content/$MODEL.trt
    echo $MODEL
    cd /content/YuShanAICompetition/csrc/deploy_gpu/bin
    ./trt_inference /content/$MODEL.trt /content/YuShanAICompetition/sample.jpg $1
done