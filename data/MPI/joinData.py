import pandas as pd
import os

def get_folder_files(folder):
    files = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            files.append(file)
    return files

def read_csv(file_path):
    return pd.read_csv(file_path)

def write_csv(data, file_path):
    data.to_csv(file_path, index=False)

def read_node_data(number: int) -> pd.DataFrame:
    folder = "Node_" + str(number)
    files = get_folder_files(folder)

    data = pd.DataFrame()
    for file in files:
        file_path = folder + "/" + file
        #  Check if the file is empty
        if os.stat(file_path).st_size == 0:
            continue
        df = read_csv(file_path)
        df.insert(0, 'Node', number)
        data = pd.concat([data, df], ignore_index=True)

    return data

# Example usage
if __name__ == "__main__":
    data_node_1 = read_node_data(1)
    data_node_2 = read_node_data(2)
    data_node_3 = read_node_data(3)

    joined_data = pd.concat([data_node_1, data_node_2, data_node_3], ignore_index=True)
    write_csv(joined_data, "mpi.csv")
