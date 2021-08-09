import torch
import os

from .model import get_pred_model
from .preprocess import preprocess
from .utils import ImageReader, int_label2word

def single_predict(image_path, model):
    test_image = ImageReader.read_and_show_image(image_path)
    test_tensor = preprocess(test_image)
    logits = F.softmax(model(test_tensor.float.to(device)))
    prediction = int(torch.argmax(logits))
    confidence = logits[prediction]
    word = int_label2word(prediction)
    print(f"Prediction: {word}, Class Number: {prediction}, confidence: {confidence}")
    return word, prediction, confidence
    

def predict(args):
    # TODO
    device = "cuda:0" if torch.cuda.is_available() and args.is_gpu_used else "cpu"
    model = get_pred_model()
    model.to(device)
    model.eval()
    print("test")
    # assert os.path.exists(args.input_path)
    # if os.path.isdir(args.input_path):
    #     for image_path in os.listdir(args.input_path):
    #         single_predict(image_path, model)
    # else: 
    #     single_predict(args.input_path, model)