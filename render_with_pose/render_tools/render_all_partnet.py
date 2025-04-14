import os
import sys
from argparse import ArgumentParser

sys.path.append('./utils')
from utils.config_utils import PARTNET_ID_PATH, PARTNET_CAMERA_POSITION_RANGE, HEIGHT, WIDTH

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ray_tracing', type=bool, default=False, help='Specify whether to use ray tracing in rendering')
    parser.add_argument('--replace_texture', type=bool, default=False, help='Specify whether to replace the texture of the rendered image using the original model')
    parser.add_argument('--start_idx', type=int, default=0, help='Specify the start index of the model id to render')
    parser.add_argument('--num_render', type=int, default=32, help='Specify the number of renderings for each model id each camera range')
    parser.add_argument('--log_dir', type=str, default='./log_render.log', help='Specify the log file')
    parser.add_argument('--choose_category', type=str, default=None, help='Specify the category to render')
    
    args = parser.parse_args()
    
    ray_tracing = args.ray_tracing
    replace_texture = args.replace_texture
    start_idx = args.start_idx
    num_render = args.num_render
    log_dir = args.log_dir

    model_id_list = []
    with open(PARTNET_ID_PATH, 'r') as fd:
        for line in fd:
            ls = line.strip().split(' ')
            if args.choose_category is None or ls[0] == args.choose_category:
                model_id_list.append((ls[0], int(ls[1])))

    total_to_render = len(model_id_list)
    cnt = 0

    for category, model_id in model_id_list:
        print(f'Still to render: {total_to_render-cnt}\n')

        for pos_idx in range(len(PARTNET_CAMERA_POSITION_RANGE[category])):
            for render_idx in range(num_render):
                print(f'Rendering: {category} : {model_id} : {pos_idx} : {start_idx + render_idx}\n')
                
                render_string = f'python -u render.py --dataset partnet --model_id {model_id} --camera_idx {pos_idx} --render_idx {start_idx + render_idx} --height {HEIGHT} --width {WIDTH}'
                if ray_tracing:
                    render_string += ' --ray_tracing True'
                if replace_texture:
                    render_string += ' --replace_texture True'
                render_string += f' 2>&1 | tee -a {log_dir}'
                
                os.system(render_string)

        print(f'Render Over: {category} : {model_id}\n')
        cnt += 1
    
    print("Over!!!")
