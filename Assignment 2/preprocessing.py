"""
Module to perform preprocessing on the data
"""
import h5py
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import time

INPUT = Path('data')
OUTPUT = Path('preprocessed_data')

def read_file(file_path: Path) -> pd.DataFrame:
    """
    Reads the file from the path, and transposes the data so the features are on the columns
    returns a Dataframe
    """
    dataset_name = get_dataset_name(file_path)
    with h5py.File(file_path, 'r') as f:
        data = f.get(dataset_name)[()]
    return pd.DataFrame(data).T

def get_dataset_name(file_name_with_dir: Path):
    """
    extracts the dataset name from the filepath
    """
    filename_without_dir = file_name_with_dir.name
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = '_'.join(temp)
    return dataset_name

def preprocess_file(filename_path: Path, scaler = StandardScaler, drop_out_factor = 10):
    """
    applies the preprocessing pipeline to 1 file
    """
    # Loading in the data
    initial = read_file(filename_path)
    #The preprocessing
    dropped_out = initial.T.iloc[:, ::drop_out_factor].T
    scaled_and_transposed = pd.DataFrame(scaler.transform(dropped_out),
                                        columns=dropped_out.columns,
                                        index = dropped_out.index)
    preprecossed = scaled_and_transposed
    #storing the preprocessed data
    path = (OUTPUT / filename_path.relative_to(INPUT)).with_suffix('.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    preprecossed.to_csv(path)

def main(data_type: str):
    """
    applies the preprocessing pipeline to all files in a dataset
    """
    if not (data_type == 'Cross' or data_type == 'Intra'):
        raise ValueError("data_type must be either 'Cross' or 'Intra'")

    # Get filepaths of all files
    files = []
    for folder in (INPUT / Path(data_type)).iterdir():
        if folder == INPUT / Path(data_type) / Path('.DS_Store'):
            print(f'Skipping: {folder} ')
            continue
        for filepath in folder.iterdir():
            files.append(filepath)

    # setup scaler
    train_files = [f for f in files if 'train' in f.parts]
    # Only fit the scaler on training data, but use it to scale all data as would be the case in a real world application
    scaler = StandardScaler()
    for file in train_files:
        df = read_file(file)
        scaler.partial_fit(df)
    print('Scaler setup done')

    #Do preprocessing per file
    for file in files:
        preprocess_file(file, scaler)
    print('Preprocessing done')


if __name__ == '__main__':
    start = time.perf_counter()
    main('Cross')
    end = time.perf_counter()
    print(f"Execution time: {end - start:.4f} seconds")

    start = time.perf_counter()
    main('Intra')
    end = time.perf_counter()
    print(f"Execution time: {end - start:.4f} seconds")
