import os
import shutil
checkpoint_root = 'checkpoint_5'
checkpoints = os.listdir(checkpoint_root)
for i in range(len(checkpoints)):
    checkpoint_idx = int(checkpoints[i].replace('checkpoint_', ''))
    if checkpoint_idx % 20:
        path = os.path.join(checkpoint_root, 'checkpoint_%06d' % checkpoint_idx)
        shutil.rmtree(path, ignore_errors=True)