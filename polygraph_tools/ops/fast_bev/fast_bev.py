# Copyright (c) Phigent Robotics. All rights reserved.

import numpy as np
import torch
from fast_bev_ext import fast_bev_forward

def fast_bev_deploy(imgs,
                    img_coors, 
                    bev_coors,
                    bev_volume):
    b, n, h, w, c = imgs.size()
    # bev_feat = imgs.new_zeros((b * bev_volume, c))
    # bs, ns, hs, ws = img_coors.split(1, dim=1)
    # feat = imgs[bs.squeeze(), ns.squeeze(), hs.squeeze(), ws.squeeze()]
    # indices = bev_coors[:,0] * bev_volume + bev_coors[:,1]
    # bev_feat.index_add_(0, indices, feat)
    bev_feat = imgs.new_zeros((b * bev_volume, c))
    fast_bev_forward(bev_volume, imgs.contiguous(), img_coors.int().contiguous(), 
                     bev_coors.int().contiguous(), bev_feat)
    return bev_feat


class TRRFastBEV(torch.autograd.Function):

    @staticmethod
    def symbolic(g,
                 imgs,
                 img_coors, 
                 bev_coors,
                 bev_volume=9600):
        """symbolic function for creating onnx op."""
        return g.op(
            'rcdeploy::fast_bev',
            imgs,
            img_coors, 
            bev_coors,
            bev_volume_i=bev_volume)

    @staticmethod
    def forward(g,
                imgs,
                img_coors, 
                bev_coors,
                bev_volume=9600):
        """run forward."""
        # for deploy
        bev_feat = fast_bev_deploy(imgs,
                                   img_coors, 
                                   bev_coors,
                                   bev_volume=bev_volume) 
        return bev_feat


