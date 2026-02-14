import torch.nn as nn
import torch
import torch as th
import numpy as np
import torch.nn.functional as F
from batchgenerators.augmentations.utils import pad_nd_image
from typing import Tuple

from basic_modules import conv_nd
from highway_network_modules import ConvDropoutNormNonlin, ConvDropoutNonlinNorm, StackedConvLayers, hwUpsample, InitWeights_He

sigmoid_helper = lambda x: F.sigmoid(x)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def get_device(self):
        if next(self.parameters()).device.type == "cpu":
            return "cpu"
        else:
            return next(self.parameters()).device.index

    def set_device(self, device):
        if device == "cpu":
            self.cpu()
        else:
            self.cuda(device)

    def forward(self, x):
        raise NotImplementedError
    
class SegmentationNetwork(NeuralNetwork):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # if we have 5 pooling then our patch size must be divisible by 2**5
        self.input_shape_must_be_divisible_by = None  # for example in a 2d network that does 5 pool in x and 6 pool
        # in y this would be (32, 64)

        # we need to know this because we need to know if we are a 2d or a 3d netowrk
        self.conv_op = None  # nn.Conv2d or nn.Conv3d

        # this tells us how many channels we have in the output. Important for preallocation in inference
        self.num_classes = None  # number of channels in the output

        # depending on the loss, we do not hard code a nonlinearity into the architecture. To aggregate predictions
        # during inference, we need to apply the nonlinearity, however. So it is important to let the newtork know what
        # to apply in inference. For the most part this will be softmax
        self.inference_apply_nonlin = lambda x: x  # softmax_helper

        # This is for saving a gaussian importance map for inference. It weights voxels higher that are closer to the
        # center. Prediction at the borders are often less accurate and are thus downweighted. Creating these Gaussians
        # can be expensive, so it makes sense to save and reuse them.
        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None

    def predict_2D(self, x, do_mirroring: bool, mirror_axes: tuple = (0, 1),
                   use_sliding_window: bool = False, step_size: float = 0.5, patch_size: tuple = None,
                   regions_class_order: tuple = None, use_gaussian: bool = False,
                   pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        torch.cuda.empty_cache()

        if self.conv_op != nn.Conv2d:
            raise RuntimeError("Model is not Conv2D")


        assert step_size <= 1, 'step_size must be smaler than 1. Otherwise there will be a gap between consecutive ' \
                               'predictions'

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        assert len(x.shape) == 3, "data must have shape (c,x,y)"

        with torch.no_grad():
            res = self._internal_predict_2D_2Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                        pad_border_mode, pad_kwargs, verbose)
        return res

    def _internal_predict_2D_2Dconv(self, x: np.ndarray, min_size: Tuple[int, int], do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1, 2), regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        
        assert len(x.shape) == 3, "x must be (c, x, y)"

        assert self.input_shape_must_be_divisible_by is not None, 'input_shape_must_be_divisible_by must be set to ' \
                                                                  'run _internal_predict_2D_2Dconv'

        data, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True,
                                    self.input_shape_must_be_divisible_by)

        predicted_probabilities = self._internal_maybe_mirror_and_pred_2D(data[None], mirror_axes, do_mirroring,
                                                                          None)[0]

        slicer = tuple(
            [slice(0, predicted_probabilities.shape[i]) for i in range(len(predicted_probabilities.shape) -
                                                                       (len(slicer) - 1))] + slicer[1:])
        predicted_probabilities = predicted_probabilities[slicer]

        if regions_class_order is None:
            predicted_segmentation = predicted_probabilities.argmax(0)
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
        else:
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
            predicted_segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[predicted_probabilities[i] > 0.5] = c

        return predicted_segmentation, predicted_probabilities

    def _internal_maybe_mirror_and_pred_2D(self, x, mirror_axes=(0, 1), do_mirroring=True, mult=None):
        assert len(x.shape) == 4, "x must be (B, C, H, W)"

        device = next(self.parameters()).device
        x = x.to(device)

        result = torch.zeros((x.shape[0], self.num_classes, *x.shape[2:]), device=device)

        preds = []

        # normal prediction
        preds.append(self.inference_apply_nonlin(self(x)))

        # mirrored predictions
        if do_mirroring:
            for axis in mirror_axes:
                flipped_x = torch.flip(x, dims=[axis + 2])  # +2 skips B,C
                pred = self.inference_apply_nonlin(self(flipped_x))
                pred = torch.flip(pred, dims=[axis + 2])  # flip back
                preds.append(pred)

            # both axes flip
            if len(mirror_axes) == 2:
                flipped_x = torch.flip(x, dims=[2, 3])
                pred = self.inference_apply_nonlin(self(flipped_x))
                pred = torch.flip(pred, dims=[2, 3])
                preds.append(pred)

        result = torch.mean(torch.stack(preds), dim=0)

        return result
    
class Generic_UNet(SegmentationNetwork):
    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, highway = False,
                 deep_supervision=False, anchor_out=False, dropout_in_localization=False,
                 final_nonlin=sigmoid_helper, weightInitializer=InitWeights_He(1e-2),
                 pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        
        super(Generic_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.anchor_out = anchor_out
        self.highway = highway

        upsample_mode = 'bilinear'
        pool_op = nn.MaxPool2d
        transpconv = nn.ConvTranspose2d
        
        if pool_op_kernel_sizes is None:
            pool_op_kernel_sizes = [(2, 2)] * num_pool
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [(3, 3)] * (num_pool + 1)

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = [1 * (num_pool + 1)]
        # for krnl in self.conv_kernel_sizes:
        #     self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            self.max_num_features = self.MAX_FILTERS_2D

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.conv_trans_blocks_a = []
        self.conv_trans_blocks_b = []
        self.td = []
        self.tu = []
        self.ffparser = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            # if d < num_pool -1 and self.highway:
            #     self.conv_trans_blocks_a.append(conv_nd(2, int(d/2 + 1) * 128, 2 **(d+5), 1))
            #     self.conv_trans_blocks_b.append(conv_nd(2, 2 **(d+5), 1, 1))
            # if d != num_pool - 1 and self.highway:
            #     self.ffparser.append(FFParser(output_features, 256 // (2 **(d+1)), 256 // (2 **(d+2))+1))

            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)
        


        # now the bottleneck.
        # determine the first stride
        first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            self.tu.append(hwUpsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            # else:
            #     self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
            #                               pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))
        
        # if self._deep_supervision:
        #     for ds in range(len(self.conv_blocks_localization)):
        #         self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
        #                                         1, 1, 0, 1, 1, seg_output_use_bias))
        # else:
        self.seg_outputs.append(conv_op(self.conv_blocks_localization[-1][-1].output_channels, num_classes,
                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(hwUpsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.conv_trans_blocks_a = nn.ModuleList(self.conv_trans_blocks_a)
        self.conv_trans_blocks_b = nn.ModuleList(self.conv_trans_blocks_b)
        self.ffparser = nn.ModuleList(self.ffparser)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            
    def forward(self, x, hs = None):
        skips = []
        seg_outputs = []
        anch_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
            if hs:
                h = hs.pop(0)
                ddims = h.size(1)
                h = self.conv_trans_blocks_a[d](h)
                h = self.ffparser[d](h)
                ha = self.conv_trans_blocks_b[d](h)
                hb = th.mean(h,(2,3))
                hb = hb[:,:,None,None]
                x = x * ha * hb
            

        x = self.conv_blocks_context[-1](x)
        emb = conv_nd(2, x.size(1), 512, 1).to(device = x.device)(x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = th.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            if self._deep_supervision:
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            if self.anchor_out and (not self._deep_supervision):
                anch_outputs.append(x)
        if not seg_outputs:
            seg_outputs.append(self.final_nonlin(self.seg_outputs[0](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        if self.anchor_out:
            return tuple([i(j) for i, j in
                                        zip(list(self.upscale_logits_ops)[::-1], anch_outputs[:-1][::-1])]),seg_outputs[-1]
                                            
        else:
            return emb, seg_outputs[-1]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
        return tmp