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

    create_dataset(images_path, dataset_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data', help='directory of the data')
    args = parser.parse_args()
    main(args)
