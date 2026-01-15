Video Action Recognition w/ R(2+1)D on UCF101 (PyTorch, Decord, NumPy)

    •	Improved action recognition performance to 59.13% Top-1 / 82.24% Top-5 video-level accuracy on UCF101 Split-1, as measured by 10-clip per-video logit averaging, by training an R(2+1)D CNN from random initialization with AMP mixed precision and torch.compile.
  
    •	Achieved 56.17% Top-1 / 80.02% Top-5 best clip-level accuracy, as measured on the UCF101 test split, by building an end-to-end pipeline with Decord video decoding, 32-frame clip sampling (stride=2) at 160×160, consistent clip-wise augmentation, warmup + cosine LR scheduling, label smoothing, and checkpoint selection.
