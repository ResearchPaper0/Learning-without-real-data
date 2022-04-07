import torch

# constant paths
ROOT_PATH = '/evaluation_pipeline/datasets/AL_3K_boucles'
OUTPUT_PATH = './outs'

# learning parameters
FROM_CHECKPOINT = False
BATCH_SIZE = 4
LR = 0.001
EPOCHS = 1000
EARLY_STOP_N = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train/test split
TEST_SPLIT = 1./3

# show dataset keypoint plot
SHOW_DATASET_PLOT = True

if __name__ == '__main__':
    print(f"""
Training with "{ROOT_PATH}" dataset folder
                 Train ratio | {1-TEST_SPLIT:.3f}
            Validation ratio | {TEST_SPLIT:.3f}

Training Process
     Restore last checkpoint | {'Yes' if FROM_CHECKPOINT else 'No' }
                  Batch size | {BATCH_SIZE}
               Learning rate | {LR:.2e}
                   Max epoch | {EPOCHS}
     Early stopping patience | {EARLY_STOP_N}
                      Device | {DEVICE}


                 Save folder | "{OUTPUT_PATH}"
Show a batch before training | {'Yes' if SHOW_DATASET_PLOT else 'No' }
""")
