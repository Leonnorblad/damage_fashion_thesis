# FasterRCNN
Fine-tune and evaluate the pytorchvision model [fasterrcnn_resnet50_fpn()](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn).

## Setup
1. Clone this repository to your local machine.
2. Navigate to the repository directory.
3. Run the following command to create the conda environment `FasterRCNN_Leon`:

```bash
conda env create -f environment.yml
```

## Notebook descriptions

### `train_fasterrcnn.ipynb`
- Fine-tuning the pytorch model `fasterrcnn_resnet50_fpn` with fashion data (from `fashion_v1` folder)
- Saves best epoch (lowest val loss), last epoch and validation loss history in `/runs` folder.

### `evaluate.ipynb`
- Computes mAP at diffrent IoU values
- Predicts on images from val loader
- Predicts on images from test folder

### `tune_thresholds.ipynb`
- Test diffrent NMS and confidence thresholds
- Produces heatmaps
- Find the thresholds with the highest F1-score

### `material_for_chi2test.ipynb`
- Computes TP, FP, FN rate to compute chi square test.
