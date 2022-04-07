import os
import shutil
import glob
import numpy as np


ROOT = 'data'
NEW_ROOT = '/model_training/'

N_TEST = 1000

file_ids = glob.glob(f'{ROOT}/X/front/*.jpg')
file_ids = np.array([f.split('/')[-1].split('.')[0] for f in file_ids])

print('Total files:', len(file_ids))

print(file_ids[:10])
print('\nShuffling ids')
np.random.shuffle(file_ids)
print(file_ids[:10])



print('\nSplit into train-val/test')
file_ids_train_val, file_ids_test = file_ids[:-N_TEST], file_ids[-N_TEST:]
print('Training files:', len(file_ids_train_val))
print('Training files:', len(file_ids_test))


print('\nCreate new directories')
TRAIN_VAL_ROOT = f'{NEW_ROOT}/train_val'
TEST_ROOT = f'{NEW_ROOT}/test'
try: os.makedirs(NEW_ROOT)
except: pass
try: os.makedirs(TRAIN_VAL_ROOT)
except: pass
try: os.makedirs(TEST_ROOT)
except: pass

try: os.makedirs(f'{TRAIN_VAL_ROOT}/X')
except: pass
try: os.makedirs(f'{TRAIN_VAL_ROOT}/Y')
except: pass
try: os.makedirs(f'{TRAIN_VAL_ROOT}/X/front')
except: pass
try: os.makedirs(f'{TRAIN_VAL_ROOT}/Y/data')
except: pass
try: os.makedirs(f'{TRAIN_VAL_ROOT}/Y/dict_points')
except: pass
try: os.makedirs(f'{TRAIN_VAL_ROOT}/Y/log')
except: pass

try: os.makedirs(f'{TEST_ROOT}/X')
except: pass
try: os.makedirs(f'{TEST_ROOT}/Y')
except: pass
try: os.makedirs(f'{TEST_ROOT}/X/front')
except: pass
try: os.makedirs(f'{TEST_ROOT}/Y/data')
except: pass
try: os.makedirs(f'{TEST_ROOT}/Y/dict_points')
except: pass
try: os.makedirs(f'{TEST_ROOT}/Y/log')
except: pass



print('\nMove files')
for ids, folder in zip([file_ids_train_val, file_ids_test], [TRAIN_VAL_ROOT, TEST_ROOT]):
    for id in ids:
        print('Moving file', id)
        os.rename(f'{ROOT}/X/front/{id}.jpg', f'{folder}/X/front/{id}.jpg')
        os.rename(f'{ROOT}/Y/data/{id}.json', f'{folder}/Y/data/{id}.json')
        os.rename(f'{ROOT}/Y/dict_points/{id}.json', f'{folder}/Y/dict_points/{id}.json')
        os.rename(f'{ROOT}/Y/log/{id}.json', f'{folder}/Y/log/{id}.json')
