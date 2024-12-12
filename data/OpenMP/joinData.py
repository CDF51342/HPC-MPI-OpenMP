import pandas as pd
import os

folder_path = os.path.dirname(os.path.abspath(__file__))

CSV_TO_FILES = {
    "ReadGray(s)": os.path.join(folder_path, "gray/time_read-pgm.csv"),
    "ReadColor(s)": os.path.join(folder_path, "color/time_read-ppm.csv"),
    "Gray(s)": os.path.join(folder_path, "gray/time_G.csv"),
    "Hsl(s)": os.path.join(folder_path, "color/time_HSL.csv"),
    "Yuv(s)": os.path.join(folder_path, "color/time_YUV.csv"),
    "WriteGray(s)": os.path.join(folder_path, "gray/time_write-pgm.csv"),
    "WriteHsl(s)": os.path.join(folder_path, "color/time_write-HSL.csv"),
    "WriteYuv(s)": os.path.join(folder_path, "color/time_write-YUV.csv")
}

def read_csv(file_path):
    return pd.read_csv(file_path)

def write_csv(data, file_path):
    data.to_csv(os.path.join(folder_path, file_path), index=False)

def confidence_interval(x):
    upperLimit = x.mean() + 1.96 * x.std() / (len(x) ** 0.5)
    lowerLimit = x.mean() - 1.96 * x.std() / (len(x) ** 0.5)
    filtered_x = x[(x >= lowerLimit) & (x <= upperLimit)]
    return filtered_x.mean()

def read_data() -> pd.DataFrame:
    values = {}
    data = pd.DataFrame()
    
    for key, value in CSV_TO_FILES.items():
        df = read_csv(value)
        values[key] = df

    for i in range(len(values["ReadGray(s)"])):
        # Create a row for the dataframe
        df = pd.DataFrame(columns=["Num Threads", "Schedule", "ChunkSize", "ReadGray(s)", "ReadColor(s)", "Gray(s)", "Hsl(s)", "Yuv(s)", "WriteGray(s)", "WriteHsl(s)", "WriteYuv(s)", "Total(s)"])

        for key in CSV_TO_FILES.keys():
            value = values[key].iloc[i]["Time (s)"]
            df[key] = [value]

        df["Num Threads"] = values["ReadGray(s)"].iloc[i]["Threads"]
        df["Schedule"] = values["ReadGray(s)"].iloc[i]["Schedule"]
        df["ChunkSize"] = values["ReadGray(s)"].iloc[i]["ChunkSize"]
        df["Total(s)"] = values["ReadGray(s)"].iloc[i]["TotalTime"]

        data = pd.concat([data, df], ignore_index=True)

    return data

def add_label_column(data: pd.DataFrame) -> pd.DataFrame:
    del data['index']
    data['Label'] = "OpenMP: Threads " + data['Num Threads'].astype(str) + ", Schedule " + data['Schedule'] + ", ChunkSize " + data['ChunkSize'].astype(str)
    cols = ['Label'] + [col for col in data if col != 'Label']
    data = data[cols]
    return data

# Example usage
if __name__ == "__main__":
    try:
        data = read_data()

        data = data.groupby(['Num Threads', 'Schedule', 'ChunkSize'], as_index=False).agg(confidence_interval).reset_index()
        data = data.round(4)  # Round to 4 decimal places

        result_data = add_label_column(data)

        write_csv(result_data, "joinedData.csv")
        print("\033[92m" + "[OK]" + "\033[0m" + " OpenMP data joined successfully")
    except Exception as e:
        print("\033[91m" + "[ERROR]" + "\033[0m" + " Error joining OpenMP data")
        print(e)
        raise