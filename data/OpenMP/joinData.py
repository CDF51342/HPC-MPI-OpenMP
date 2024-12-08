import pandas as pd
import os

CSV_TO_FILES = {
    "ReadGray(s)": "gray/time_read-pgm.csv",
    "ReadColor(s)": "color/time_read-ppm.csv",
    "Gray(s)": "gray/time_G.csv",
    "Hsl(s)": "color/time_HSL.csv",
    "Yuv(s)": "color/time_YUV.csv",
    "WriteGray(s)": "gray/time_write-pgm.csv",
    "WriteHsl(s)": "color/time_write-HSL.csv",
    "WriteYuv(s)": "color/time_write-YUV.csv"
}

def read_csv(file_path):
    return pd.read_csv(file_path)

def write_csv(data, file_path):
    data.to_csv(file_path, index=False)

def read_data() -> pd.DataFrame:
    values = {}
    data = pd.DataFrame()
    
    for key, value in CSV_TO_FILES.items():
        df = read_csv(value)
        values[key] = df

    for i in range(len(values["ReadGray(s)"])):
        # Create a row for the dataframe
        df = pd.DataFrame(columns=["Node", "Processes", "Num Threads", "Schedule", "ChunkSize", "ReadGray(s)", "ReadColor(s)", "Gray(s)", "Hsl(s)", "Yuv(s)", "WriteGray(s)", "WriteHsl(s)", "WriteYuv(s)", "Total(s)"])

        df["Node"] = [1]
        df["Processes"] = [1]

        for key in CSV_TO_FILES.keys():
            value = values[key].iloc[i]["Time (s)"]
            df[key] = [value]

        df["Num Threads"] = values["ReadGray(s)"].iloc[i]["Threads"]
        df["Schedule"] = values["ReadGray(s)"].iloc[i]["Schedule"]
        df["ChunkSize"] = values["ReadGray(s)"].iloc[i]["ChunkSize"]
        df["Total(s)"] = values["ReadGray(s)"].iloc[i]["TotalTime"]

        data = pd.concat([data, df], ignore_index=True)

    return data

# Example usage
if __name__ == "__main__":
    data = read_data()

    write_csv(data, "openMP.csv")