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

class UNetQuarter(nn.Module):
    def __init__(self, n_channels, n_classes, max_chans=64):
        super(UNetQuarter, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.max_chans = max_chans

        self.inc = (DoubleConv(n_channels, 64))        
        self.down_pre1 = (DownAvg(64, 64))
        self.down_pre2 = (DownAvg(64, max_chans))
        self.down1 = (Down(max_chans, max_chans))
        self.down2 = (Down(max_chans, max_chans))
        self.down3 = (Down(max_chans, max_chans))
        self.down4 = (Down(max_chans, max_chans))
        self.up1 = (Up(max_chans*2, max_chans, True))
        self.up2 = (Up(max_chans*2, max_chans, True))
        self.up3 = (Up(max_chans*2, max_chans, True))
        self.up4 = (Up(max_chans*2, max_chans, True))
        self.outc = (OutConv(max_chans, n_classes))

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

class UNetBlocks(nn.Module):
    def __init__(self, n_channels, n_classes, 
                 max_chans=64, pre_merge = False, post_merge = False):
        super(UNetBlocks, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.max_chans = max_chans
        self.pre_merge = pre_merge
        self.post_merge = post_merge

        self.inc = (DoubleConv(n_channels, 64))        
        self.down_pre1 = (DownAvg(64, 64))
        self.down_pre2 = (DownAvg(64, max_chans))
        if self.pre_merge:
            self.merge_input = Merge(max_chans)
        self.down1 = (Down(max_chans, max_chans))
        self.down2 = (Down(max_chans, max_chans))
        self.down3 = (Down(max_chans, max_chans))
        self.down4 = (Down(max_chans, max_chans))
        self.up1 = (Up(max_chans*2, max_chans, True))
        self.up2 = (Up(max_chans*2, max_chans, True))
        self.up3 = (Up(max_chans*2, max_chans, True))
        self.up4 = (Up(max_chans*2, max_chans, True))
        if self.post_merge:
            self.merge_output = Merge(max_chans)        
        self.outc = (OutConv(max_chans, n_classes))

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


class UNetTrack(nn.Module):
    ''' This takes in heatmap and/or image from previous frame
    '''
    def __init__(self, add_prev_im, add_prev_out, n_classes, 
                 max_chans=64):
        super(UNetTrack, self).__init__()
        self.add_prev_im = add_prev_im
        self.add_prev_out = add_prev_out
        self.n_channels = 3 
        if self.add_prev_im:
            self.n_channels += 3
        self.n_classes = n_classes
        self.max_chans = max_chans

        self.inc = (DoubleConv(self.n_channels, 64))        
        self.down_pre1 = (DownAvg(64, 64))
        self.down_pre2 = (DownAvg(64, max_chans))
        if self.add_prev_out:
            self.add = Add( max_chans )
        self.down1 = (Down(max_chans, max_chans))
        self.down2 = (Down(max_chans, max_chans))
        self.down3 = (Down(max_chans, max_chans))
        self.down4 = (Down(max_chans, max_chans))
        self.up1 = (Up(max_chans*2, max_chans, True))
        self.up2 = (Up(max_chans*2, max_chans, True))
        self.up3 = (Up(max_chans*2, max_chans, True))
        self.up4 = (Up(max_chans*2, max_chans, True))
        self.outc = (OutConv(max_chans, n_classes))

    def forward(self, xlist):
        x = self.inc(xlist[0])
        x = self.down_pre1(x)
        x1 = self.down_pre2(x)
        if self.add_prev_out:
            x1 = self.add( x1, xlist[1] )
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

    def apply_to_stack(self, imstack, Nmax=None):
        Nim=imstack.shape[1]//3
        if not Nmax is None:
            N = min(Nim-1,Nmax) if self.add_prev_im else min(Nim,Nmax)
        else:
            N = Nim-1 if self.add_prev_im else Nim
        xout = None    
        for i in range(N,0,-1):
            if self.add_prev_im:
                im_input = [imstack[:,i*3-3:(i+1)*3,...]]
            else:
                im_input = [imstack[:,i*3-3:i*3,...]]
            if self.add_prev_out:
                im_input.append(xout)
            xout = self.forward(im_input)
        return xout


