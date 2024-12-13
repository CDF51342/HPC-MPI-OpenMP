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
    filtered_x = x
    return filtered_x.mean()

def read_data() -> pd.DataFrame:
    values = {}    
    for key, value in CSV_TO_FILES.items():
        df = read_csv(value)
        values[key] = df

    data = pd.DataFrame(columns=["ReadGray(s)", "ReadColor(s)", "Gray(s)", "Hsl(s)", "Yuv(s)", "WriteGray(s)", "WriteHsl(s)", "WriteYuv(s)", "Total(s)"])

    # Instead of put each value as bottom, we will put the mean value
    for key in CSV_TO_FILES.keys():
        value = confidence_interval(values[key]["Time (s)"])
        data[key] = [value]

    data["Total(s)"] = confidence_interval(values["ReadGray(s)"]["TotalTime"])
    data = data.round(4)  # Round to 4 decimal places

    return data

def add_label_column(data: pd.DataFrame) -> pd.DataFrame:
    data['Label'] = "Sequential"
    cols = ['Label'] + [col for col in data if col != 'Label']
    data = data[cols]
    return data

# Example usage
if __name__ == "__main__":
    try:
        data = read_data()

        result_data = add_label_column(data)

        write_csv(result_data, "joinedData.csv")
        print("\033[92m" + "[OK]" + "\033[0m" + " OpenMP data joined successfully")
    except Exception as e:
        print("\033[91m" + "[ERROR]" + "\033[0m" + " Error joining OpenMP data")
        print(e)
        raise