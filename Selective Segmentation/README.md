Temperature-Scaled Uncertainty-Aware Selective Segmentation (MC Dropout) — PyTorch, Colab Pro

Computer Vision | Segmentation Reliability & Calibration

	•	Improved segmentation reliability as measured by selective risk dropping from 0.132 → 0.024 (≈82% reduction) at ≈0.60 coverage, by abstaining on high-uncertainty pixels using predictive entropy from MC Dropout (T=20 samples).
	
	•	Increased usable segmentation quality as measured by selective mIoU improving from 0.677 (full coverage) to a peak of 0.789 at ≈0.70 coverage, by ranking pixels via uncertainty quantiles and evaluating fixed-coverage selective mIoU.
	
	•	Improved probability calibration as measured by test pixel-ECE reducing from 0.0098 → 0.0084, by fitting post-hoc temperature scaling (T* = 1.0104) on the validation set using pixel-wise NLL.
	
	•	Achieved strong baseline segmentation performance as measured by best validation mIoU = 0.6769, by training a U-Net (7.76M params) with AdamW + AMP and a CE+Dice objective on Oxford-IIIT Pet trimap segmentation.
	
	•	Reduced GPU memory overhead for uncertainty estimation as measured by stable inference without OOM, by implementing streaming MC aggregation (no T×B×C×H×W stacking) and separate small-batch calibration loaders.
