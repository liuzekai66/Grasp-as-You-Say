
from .ldgd import DDPM
from .irf import RenfinmentTransformer
def build_model(cfg):
    if cfg.name.lower() == "ldgd":
        return DDPM(cfg)
    elif cfg.name.lower() == "irf":
        return RenfinmentTransformer(cfg)

    else:
        raise Exception("1")
