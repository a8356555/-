def hook_fn(m, i, o):
    """Used for gradient check"""
    print(m)
    print("------------Input Grad------------")

    for grad in i:
        try:
            print(grad.shape)
        except AttributeError:
            print ("None found for Gradient")

    print("------------Output Grad------------")
    for grad in o:
        try:
            print(grad.shape)
        except AttributeError:
            print ("None found for Gradient")
    print("\n")

# for name, layer in model.model._modules.items():
#     if 'layer' in name:
#         for name, layer in layer._modules.items():
#             for name, layer in layer._modules.items():
#                 layer.register_backward_hook(hook_fn)
#     else:
#         layer.register_backward_hook(hook_fn)



# profiler
def pytorch_profiler(model, criterion, optimizer, datamodule):    
    def trace_handler(prof):
            print(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))
        
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
                    ],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=0,
            active=1),
        on_trace_ready=trace_handler
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    ) as profiler:
            # for step, data in enumerate(train_dataloader, 0):
            for step, data in enumerate(data_module.train_dataloader()):
                print("step:{}".format(step))
                # inputs, labels = data[0].to(dcfg.device), data[1].to(dcfg.device)
                inputs, labels = data[0]['data'].to(dcfg.device), data[0]['label'].to(dcfg.device)
                model = model.to(dcfg.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long().squeeze(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                profiler.step()