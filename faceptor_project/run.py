
import os
import torch
import argparse
from PIL import Image
from torch import distributed
import numpy as np

from core.data.transform.attribute_analysis_transform import attribute_test_transform
from core.utils import printlog
from core.config import Config
from core.solver import solver_entry

try:
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    distributed.init_process_group('gloo')
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend='gloo',
        init_method='tcp://127.0.0.1:12584',
        rank=rank,
        world_size=world_size,
    )

# for Faceptor-Base Stage-1
# args = argparse.Namespace(
#   config='./faceptor_base_affect.yaml',
#   expname='train_recog_age_biattr_affect_parsing_align',  # Locked
#   load_ignore=[],
#   load_iter='50000',  # Locked
#   load_path='',
#   now='20250514_174749',
#   out_dir='./output/faceptor/stage_1/',  # Locked
#   start_time='20231223_234836',  # Locked
# )

# # for Faceptor-Full Stage-1
# args = argparse.Namespace(
#   config='./faceptor_full_affect.yaml',
#   expname='train_recog_age_biattr_affect_parsing_align_plus',  # Locked
#   load_ignore=[],
#   load_iter='50000',  # Locked
#   load_path='',
#   now='20250514_174749',
#   out_dir='./output/faceptor/stage_1/',  # Locked
#   start_time='20240103_164523',  # Locked
# )

# for naive-Faceptor
args = argparse.Namespace(
  config='./native_faceptor_affect.yaml',
  expname='train_recog_age_biattr_affect_parsing_align',  # Locked
  load_ignore=[],
  load_iter='50000',  # Locked
  load_path='',
  now='20250514_174749',
  out_dir='./output/naive_faceptor/',  # Locked
  start_time='20231208_230635',  # Locked
)

printlog('args:', args)

C = Config(args, rank, local_rank, world_size)
print(f'C.config: {C.config}')
S = solver_entry(C)

# S.test()
S.create_model()
S.load(load_items=['state_dict', 'step'])
S.create_evaluators()
S.model.set_mode_to_evaluate()
evaluator = S.evaluators['affect_rafdb']
S.model.set_evaluation_task('affect_rafdb')

# evaluator(S.last_iter, S.model)
S.model.to('cpu')
S.model.eval()

# evaluator.ver_test(S.model, S.last_iter)
img = Image.open('./data/RAF-DB/basic/data/test_0001_aligned.jpg').convert('RGB')
transform = attribute_test_transform()
img = transform(img).unsqueeze_(0)

# input = {'image': img.cuda()}
input = {'image': img.cpu()}
output = S.model(input)

S.model.train()
torch.cuda.empty_cache()
