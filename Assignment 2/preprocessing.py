import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

INPUT = Path('data')
OUTPUT = Path('preprocessed_data')
def get_dataset_name(file_name_with_dir: Path):

    filename_without_dir = file_name_with_dir.name
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = '_'.join(temp)
    return dataset_name

def preprocess_file(filename_path: Path, scaler = StandardScaler, drop_out_factor = 10):
    # Loading in the data
    dataset_name = get_dataset_name(filename_path)
    with h5py.File(filename_path, 'r') as f:
        matrix = f.get(dataset_name)[()]
    # Actual preprocessing
    initial = pd.DataFrame(matrix)
    dropped_out = initial.iloc[:, ::drop_out_factor]
    scaled_and_transposed = pd.DataFrame(scaler.transform(dropped_out.T),
                                        columns=dropped_out.T.columns,
                                        index = dropped_out.T.index)
    preprecossed = scaled_and_transposed.T
    #storing the preprocessed data
    path = (OUTPUT / filename_path.relative_to(INPUT)).with_suffix('.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    preprecossed.to_csv(path)



def main(data_type: str):
    if data_type != 'Cross' or data_type != 'Intra':
        raise Exception()

    # Get filepaths of all files
    files = []
    for folder in (INPUT / Path(data_type)).iterdir():
        if folder == INPUT / Path(data_type) / Path('.DS_Store'):
            print(folder)
            continue
        for filepath in folder.iterdir():
            files.append(filepath)

    # setup scaler
    scaler = StandardScaler()
    for file in files:
        dataset_name = get_dataset_name(file)
        with h5py.File(file, 'r') as f:
            matrix = pd.DataFrame(f.get(dataset_name)[()]).T
            scaler.partial_fit(matrix)

    print('scaler setup done')
    #Do preprocessing per file
    for file in files:
        dataset_name = get_dataset_name(file)
        with h5py.File(file, 'r') as f:
            matrix = f.get(dataset_name)[()]
            preprocess_file(file, scaler)

    print('Preprocessing done')


if __name__ == '__main__':
    main('Cross')
    print('all done')
