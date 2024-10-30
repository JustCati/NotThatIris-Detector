import os
import argparse

from src.utils.dataset_utils.drct import generate_lq_images, create_dataset




def main(args):
    data_path = args.path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'{data_path} does not exist')

    iris_path = os.path.join(data_path, 'Iris-Thousand')
    images_path = os.path.join(iris_path, 'images')
    dataset_path = os.path.join(iris_path, 'data')

    SCALE_FACTOR = args.down_scale_factor
    hq_path = create_dataset(images_path, dataset_path)
    generate_lq_images(hq_path, SCALE_FACTOR)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data', help='directory of the data')
    parser.add_argument('--down_scale_factor', type=int, default=4, help='down scale factor for generating low quality images')
    args = parser.parse_args()
    main(args)
