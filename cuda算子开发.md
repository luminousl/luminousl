调用flashattention之前的辅助函数。
1. calCuQCuKVSeqLensAndKVEndIdxsKernel
核心作用：计算元数据偏移（Metadata Prep）
该 Kernel 主要负责计算用于 变长序列处理（Var-len/Packed sequence） 的索引数组。在推断时，不同请求的序列长度不同，FMHA 需要知道每个序列在内存中的起始和结束位置。
特点：这是一个非常轻量级的 Kernel（仅 1 个线程块，1 个线程执行），本质上是在 GPU 上执行一个小的串行循环，避免了将数据拷贝回 CPU 计算再拷贝回 GPU 的延迟。
2. cvtKVCachelayoutXQAToFMHAKernel
核心作用：张量布局转换（Layout Permutation）
这个 Kernel 执行的是 维度转置（Transpose/Permute）。它将 KV Cache 从 XQA（一种推断优化格式） 布局转换为 FMHA（Flash Multi-Head Attention） 期望的布局。

