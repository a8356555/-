
def img_tensor2numpy(tensor):
    return tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()

