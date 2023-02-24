""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class UNetSmall(nn.Module):
    def __init__(self, n_channels, n_classes, max_chans=64):
        super(UNetSmall, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.max_chans = max_chans

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, max_chans))
        self.down2 = (Down(max_chans, max_chans))
        self.down3 = (Down(max_chans, max_chans))
        self.down4 = (Down(max_chans, max_chans))
        self.up1 = (Up(max_chans*2, max_chans, True))
        self.up2 = (Up(max_chans*2, max_chans, True))
        self.up3 = (Up(max_chans*2, max_chans, True))
        self.up4 = (Up(64 + max_chans, 64, True))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class UNetSmallQuarter(nn.Module):
    def __init__(self, n_channels, n_classes, max_chans=64):
        super(UNetSmallQuarter, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.max_chans = max_chans

        self.inc = (DoubleConv(n_channels, 64))        
        self.down_pre1 = (DownAvg(64, 64))
        self.down_pre2 = (DownAvg(64, 64))
        self.down1 = (Down(64, max_chans))
        self.down2 = (Down(max_chans, max_chans))
        self.down3 = (Down(max_chans, max_chans))
        self.down4 = (Down(max_chans, max_chans))
        self.up1 = (Up(max_chans*2, max_chans, True))
        self.up2 = (Up(max_chans*2, max_chans, True))
        self.up3 = (Up(max_chans*2, max_chans, True))
        self.up4 = (Up(64 + max_chans, 64, True))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x = self.inc(x)
        x = self.down_pre1(x)
        x1 = self.down_pre2(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down_pre1 = torch.utils.checkpoint(self.down_pre1)
        self.down_pre2 = torch.utils.checkpoint(self.down_pre2)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNetBlocks(nn.Module):
    def __init__(self, n_channels, n_classes, 
                 max_chans=64, pre_merge = False, post_merge = False):
        super(UNetSmallQuarter, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.max_chans = max_chans
        self.pre_merge = pre_merge
        self.post_merge = post_merge

        self.inc = (DoubleConv(n_channels, 64))        
        self.down_pre1 = (DownAvg(64, 64))
        self.down_pre2 = (DownAvg(64, 64))
        if self.pre_merge:
            self.merge_input = Merge(64)
        self.down1 = (Down(64, max_chans))
        self.down2 = (Down(max_chans, max_chans))
        self.down3 = (Down(max_chans, max_chans))
        self.down4 = (Down(max_chans, max_chans))
        self.up1 = (Up(max_chans*2, max_chans, True))
        self.up2 = (Up(max_chans*2, max_chans, True))
        self.up3 = (Up(max_chans*2, max_chans, True))
        self.up4 = (Up(64 + max_chans, 64, True))
        if self.post_merge:
            self.merge_output = Merge(64)        
        self.outc = (OutConv(64, n_classes))

    def forward(self, xlist):
        x = self.inc(xlist[0])
        x = self.down_pre1(x)
        x1 = self.down_pre2(x)
        if self.pre_merge:
            x1 = self.merge_input(x1, xlist[1])
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.post_merge:
            x = self.merge_output(x, xlist[1])
        logits = self.outc(x)
        return [logits, x]

    def apply_to_stack(self, imstack, Nmax=None):
        N=imstack.shape[1]//3
        if not Nmax is None:
            N = min(N,Nmax)
        xout = [None, None]
        for i in range(N,0,-1):
            im_input = [imstack[:,i*3-3:i*3,...], xout[1]]
            xout = self.forward(im_input)
        return xout[0]


    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down_pre1 = torch.utils.checkpoint(self.down_pre1)
        self.down_pre2 = torch.utils.checkpoint(self.down_pre2)
        if self.pre_merge:
            self.merge_input = torch.utils.checkpoint(self.merge_input)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        if self.post_merge:
            self.merge_output = torch.utils.checkpoint(self.merge_output)
        self.outc = torch.utils.checkpoint(self.outc)

