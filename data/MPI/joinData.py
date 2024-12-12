import pandas as pd
import os

folder_path = os.path.dirname(os.path.abspath(__file__))

def get_folder_files(folder):
    files = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            files.append(file)
    return files

def read_csv(file_path):
    return pd.read_csv(file_path)

def write_csv(data, file_path):
    data.to_csv(os.path.join(folder_path, file_path), index=False)

# Calculate the 95% confidence interval for each group
def confidence_interval(x):
    upperLimit = x.mean() + 1.96 * x.std() / (len(x) ** 0.5)
    lowerLimit = x.mean() - 1.96 * x.std() / (len(x) ** 0.5)
    filtered_x = x[(x >= lowerLimit) & (x <= upperLimit)]
    return filtered_x.mean()

def read_node_data(number: int) -> pd.DataFrame:
    folder = os.path.join(folder_path, "Node_" + str(number))
    files = get_folder_files(folder)

    data = pd.DataFrame()
    for file in files:
        file_path = os.path.join(folder_path, folder, file)
        if os.stat(file_path).st_size == 0:
            continue
        row = read_csv(file_path)
        row.insert(0, "Nodes", number)
        data = pd.concat([data, row], ignore_index=True)

    data = data.groupby(['Nodes', 'Processes'], as_index=False).agg(confidence_interval).reset_index()
    data = data.round(4)  # Round to 4 decimal places
    return data

def add_label_column(data: pd.DataFrame) -> pd.DataFrame:
    del data['index']
    data['Label'] = "MPI: N " + data['Nodes'].astype(str) + ", Prc " + data['Processes'].astype(str)
    cols = ['Label'] + [col for col in data if col != 'Label']
    data = data[cols]
    return data

if __name__ == "__main__":
    try:
        folder_path = os.path.dirname(os.path.abspath(__file__))
        data = []

        node_folders = [f for f in os.listdir(folder_path) if f.startswith("Node_") and os.path.isdir(os.path.join(folder_path, f))]
        for node_folder in node_folders:
            node_number = int(node_folder.split("_")[1])
            data.append(read_node_data(node_number))

        joined_data = pd.concat(data, ignore_index=True)

        result_data = add_label_column(joined_data)

        write_csv(result_data, "joinedData.csv")
        print("\033[92m" + "[OK]" + "\033[0m" + " MPI data joined successfully")
    except Exception as e:
        print("\033[91m" + "[ERROR]" + "\033[0m" + " Error joining MPI data")
        print(e)
        raise
