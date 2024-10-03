import pytorch_lightning as pl


class BoringDenoisingModule(pl.LightningModule):
    """
    Boring Denoising Module: Just for testing purposes
    """
    def __init__(self):
        super().__init__()

    def forward(self, x_t, I_in, t):
        return x_t
