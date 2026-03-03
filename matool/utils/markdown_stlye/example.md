# 模型分析摘要

模型名称 dynamic_model_concat.onnx.engine

## 基本指标

- 模型吞吐量: **472.884**
- 模型延迟: **2.02649**


## 模型输入
|    | 名称              | shape               | format     |
|---:|:------------------|:--------------------|:-----------|
|  0 | points_feature    | [1, 12, 800, 192]   | FP32 NCHW  |
|  1 | imgs              | [1, 2, 3, 256, 960] | FP32 NCHW  |
|  2 | depth             | [1, 2, 1, 256, 960] | FP32 NCHW  |
|  3 | img_aug_matrix    | [1, 2, 4, 4]        | FP32 NCHW  |
|  4 | camera2ego        | [1, 2, 4, 4]        | FP32 NCHW  |
|  5 | camera_intrinsics | [1, 2, 4, 4]        | FP32 NCHW  |
|  6 | ranks_depth       | [1]                 | INT32 NCHW |
|  7 | ranks_feat        | [1]                 | INT32 NCHW |
|  8 | ranks_bev         | [1]                 | INT32 NCHW |
|  9 | interval_starts   | [1]                 | INT32 NCHW |
| 10 | interval_lengths  | [1]                 | INT32 NCHW |

## 模型输出
|    | 名称   | shape           | format    |
|---:|:-------|:----------------|:----------|
|  0 | cls_0  | [1, 1, 24, 100] | FP32 NCHW |
|  1 | reg_1  | [1, 2, 24, 100] | FP32 NCHW |
|  2 | reg_0  | [1, 2, 24, 100] | FP32 NCHW |
|  3 | cls_1  | [1, 1, 24, 100] | FP32 NCHW |

## 算子种类统计
|    | type                |   count |
|---:|:--------------------|--------:|
|  0 | CaskDeconvolutionV2 |       1 |
|  1 | Convolution         |     100 |
|  2 | Identity            |       2 |
|  3 | Myelin              |       1 |
|  4 | PluginV2            |       1 |
|  5 | PointWise           |       4 |
|  6 | Reformat            |      15 |
|  7 | Resize              |       2 |
|  8 | Shuffle             |       5 |
|  9 | SoftMax             |       1 |
| 10 | TopK                |       2 |

## 算子延迟top5

|    | Name                                           | type        |   平均延迟ms |   延迟占比% |
|---:|:-----------------------------------------------|:------------|-------------:|------------:|
| 62 | Conv_222 + Relu_223                            | Convolution |       0.1048 |        4.15 |
| 61 | Conv_140 + Relu_141                            | Convolution |       0.0705 |        2.79 |
|  2 | Conv_145 + Relu_146                            | Convolution |       0.0572 |        2.26 |
| 11 | Conv_63 + Add_64 + Relu_65                     | Convolution |       0.0563 |        2.23 |
| 81 | Transpose_258 + (Unnamed Layer* 229) [Shuffle] | Shuffle     |       0.0533 |        2.11 |

## 内存读写top5

|    | Name                                                          | type        |   内存读写 |
|---:|:--------------------------------------------------------------|:------------|-----------:|
| 11 | Conv_63 + Add_64 + Relu_65                                    | Convolution |   11805696 |
|  5 | Reformatting CopyNode for Input Tensor 0 to Conv_1 + Relu_2   | Reformat    |    9216000 |
|  8 | Conv_61 + Relu_62                                             | Convolution |    7873536 |
|  3 | Reformatting CopyNode for Input Tensor 0 to Conv_59 + Relu_60 | Reformat    |    7372800 |
| 79 | onnx::Concat_987 copy                                         | Reformat    |    7326720 |
