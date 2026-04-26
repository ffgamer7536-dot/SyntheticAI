import segmentation_models_pytorch as smp

def create_model(arch="deeplabv3plus", backbone="resnet50", weights="imagenet", in_channels=3, num_classes=10):
    if arch.lower() == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif arch.lower() == "unet":
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
        )
    else:
        raise ValueError(f"Architecture {arch} not supported.")
    
    return model
