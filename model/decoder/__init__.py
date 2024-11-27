from .unet import UNetModel

def build_decoder(decoder_cfg):
    if decoder_cfg.name.lower() == "unet":
        return UNetModel(decoder_cfg)
    else:
        raise NotImplementedError(f"No such decode: {decoder_cfg.name}")
