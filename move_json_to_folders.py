import os


def move_json_to_folders_fun():
    for folder in ['train', 'test', 'val']:
        for file in os.listdir(os.path.join(folder, 'images')):
            filename = file.split('.')[0] + '.' + file.split('.')[1] + '.json'
            existing_filepath = os.path.join('labels', filename)
            if os.path.exists(existing_filepath):
                new_filepath = os.path.join(folder, 'labels', filename)
                os.replace(existing_filepath, new_filepath)