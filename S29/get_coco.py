import os, shutil, json
import pickle, argparse

"""Unzip the data and and save it as a pickle file."""

def make_pkl(data_dir, dataset_json, train_flag=False):
    coco_data_list = []
    for i, data in enumerate(dataset_json['annotations']):
        image_id = data['image_id']
        caption  = data['caption']
        for img in dataset_json['images']:
            if img['id'] == image_id:
                image_url = img['coco_url']
                file_name = img['file_name']
                break
        coco_data_list.append({'image_id': image_id,'image_url': image_url, 'file_name': file_name, 'caption': caption})
    if train_flag:
        with open(os.path.join(data_dir, f'coco_train.pkl'), 'wb') as f:
            pickle.dump(coco_data_list, f)
    else:
        with open(os.path.join(data_dir, f'coco_val.pkl'), 'wb') as f:
            pickle.dump(coco_data_list, f)


def main(coco_path, data_dir):
    coco_dir = os.path.dirname(coco_path)
    # shutil.unpack_archive(coco_path, coco_dir)
    with open(os.path.join(coco_dir, 'annotations/captions_train2017.json')) as f:
        coco_train_dataset = json.load(f)
    with open(os.path.join(coco_dir, 'annotations/captions_val2017.json')) as f:
        coco_val_dataset = json.load(f)
    make_pkl(data_dir, coco_train_dataset, train_flag=True)
    # make_pkl(data_dir, coco_val_dataset)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', type=str, default='coco.zip')
    parser.add_argument('--data_dir', type=str, default='data')
    args = parser.parse_args()
    main(args.coco_path, args.data_dir)
