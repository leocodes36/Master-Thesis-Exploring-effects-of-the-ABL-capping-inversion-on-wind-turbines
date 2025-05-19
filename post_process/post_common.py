from pathlib import Path
import pandas as pd
import numpy as np
import ast
import re
import argparse

def load_study_names(data_path):
    return sorted({f.name.split('_study_')[0] for f in data_path.glob("*_study_*.stats")})

def select_study(study_names):
    print("Available studies:")
    for idx, name in enumerate(study_names, 1):
        print(f"{idx}. {name}")
    choice = int(input("\nSelect study: "))
    return study_names[choice - 1]

def list_stats_files(study, data_path):
    return sorted(data_path.glob(f"{study}_study_ws_*.stats"))

def list_hist_files(study, data_path):
    return sorted(data_path.glob(f"{study}_study_ws_*.hist"))

def list_iec_stats_files(data_path, selected_study="alpha"):
    return sorted(data_path.glob(f"{selected_study}_study_ws_*_ti_0.16_shear_0.2.stats"))

def list_iec_hist_files(data_path, selected_study="alpha"):
    return sorted(data_path.glob(f"{selected_study}_study_ws_*_ti_0.16_shear_0.2.hist"))

def parse_stats_file(stats_file):
    with open(stats_file, "r") as file:
        lines = [line.strip() for line in file if line.strip()]
    columns = lines[0].split()
    stats = {
        line.split()[0].rstrip(":"): np.array(line.split()[1:], dtype=np.float64)
        for line in lines[1:]
    }
    stats_df_raw = pd.DataFrame(stats, index=columns).rename_axis("channel").reset_index()
    stats_df_raw["channel"] = stats_df_raw["channel"].str.replace("Ch_", "", regex=True).astype(int)
    stats_df_raw.columns = stats_df_raw.columns.str.lower()
    return stats_df_raw

def parse_all_stats(files):
    stats_list = []
    for file in files:
        print(f"Parsing: {file.name}")
        file_info = parse_filename_parameters(file.stem)
        df = parse_stats_file(file)
        df = df.filter(["channel", "mean", "stdev", "quant_0.01", "quant_0.25", "quant_0.5", "quant_0.75", "quant_0.9", "quant_0.99", "del_1e+07_3", "del_1e+07_10"])
        df = df.assign(**file_info["params"])
        df = df[list(file_info["params"].keys()) + [col for col in df.columns if col not in file_info["params"]]]
        stats_list.append(df)

    if stats_list:
        return pd.concat(stats_list, ignore_index=True)
    else:
        return pd.DataFrame()
    
def parse_iec_stats(files, selected_study="alpha"):
    iec_stats_list = []
    for file in files:
        print(f"Parsing: {file.name}")
        file_info = parse_filename_parameters(file.stem)
        df = parse_stats_file(file)
        df = df.filter(["channel", "mean", "stdev", "quant_0.01", "quant_0.25", "quant_0.5", "quant_0.75", "quant_0.9", "quant_0.99", "del_1e+07_3", "del_1e+07_10"])
        df = df.assign(**file_info["params"])
        df = df[list(file_info["params"].keys()) + [col for col in df.columns if col not in file_info["params"]]]
        iec_stats_list.append(df)

    if iec_stats_list:
        return pd.concat(iec_stats_list, ignore_index=True)
    else:
        return pd.DataFrame()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Channel stats visualizer")
    parser.add_argument("--study", type=str, help="Name of the study")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode")
    return parser.parse_args()

def read_hist_lines(hist_file: Path) -> list[str]:
    with open(hist_file, "r") as file:
        return [line.strip() for line in file if line.strip()]

def extract_tag_info(tag_line: str):
    pattern = r"([a-zA-Z]+)_([\d\.]+)"
    matches = re.findall(pattern, tag_line)
    if matches:
        keys, values = zip(*matches)
        converted_values = [float(v) if "." in v else int(v) for v in values]
        return list(converted_values), list(keys)
    return None, None

def parse_pdf_line(line: str):
    try:
        range_part, prob_part = line.split(":")
        bins = list(map(float, range_part.strip().split(" - ")))
        probability = float(prob_part.strip())
        return [bins[0], bins[1], bins[2], probability]
    except:
        return None

def extract_tags_and_pdfs(lines: list[str], selected_study: str):
    tags, tag_keys, pdfs, current_pdf = [], [], [], []
    for line in lines:
        if line.startswith(selected_study):
            if current_pdf:
                pdfs.append(np.array(current_pdf))
                current_pdf = []
            tag_data, keys = extract_tag_info(line)
            if tag_data:
                tags.append(tag_data)
                if not tag_keys:
                    tag_keys = keys
        else:
            parsed_line = parse_pdf_line(line)
            if parsed_line:
                current_pdf.append(parsed_line)
    if current_pdf:
        pdfs.append(np.array(current_pdf))
    return tags, tag_keys, pdfs

def parse_hist_file(hist_file: Path, selected_study: str) -> pd.DataFrame:
    lines = read_hist_lines(hist_file)
    tags, tag_keys, pdfs = extract_tags_and_pdfs(lines, selected_study)
    df = pd.DataFrame(tags, columns=tag_keys)
    df["pdf"] = pdfs
    return df

def parse_all_hist(files, selected_study):
    hist_list = []
    for file in files:
        print(f"Parsing: {file.name}")
        df = parse_hist_file(file, selected_study)
        hist_list.append(df)

    if hist_list:
        return pd.concat(hist_list, ignore_index=True)
    else:
        return pd.DataFrame()
    
def parse_iec_hist(files, selected_study="alpha"):
    iec_hist_list = []
    for file in files:
        print(f"Parsing: {file.name}")
        df = parse_hist_file(file, selected_study)
        iec_hist_list.append(df)

    if iec_hist_list:
        return pd.concat(iec_hist_list, ignore_index=True)
    else:
        return pd.DataFrame()

def parse_filename_parameters(filename_stem):
    parts = filename_stem.split('_study_')
    if len(parts) != 2:
        return {"studyname": filename_stem, "params": {}}

    studyname, rest = parts
    tokens = rest.split('_')
    params = {}

    if len(tokens) % 2 != 0:
        raise ValueError(f"Filename parameter string is malformed: {rest}")

    for i in range(0, len(tokens), 2):
        key = tokens[i]
        val_str = tokens[i + 1]
        # Attempt to convert to int or float
        try:
            val = int(val_str)
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str  # leave as string if not numeric
        params[key] = val

    return {"studyname": studyname, "params": params}

def filter_dataframe(df, iec_df, filter_cols=None, user_input=True):
    """
    Filter a DataFrame based on selected columns and user-specified values.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        filter_cols (list or None): Which columns to offer for filtering. If None, infer all columns with basic types.
        user_input (bool): If True, asks for input via terminal. If False, expects `filter_cols` to be a dict of {col: value}.
    
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """

    # Auto-detect filterable columns if not provided
    if filter_cols is None or user_input:
        possible_cols = df.select_dtypes(include=["number", "object"]).columns.tolist()
        static_cols = ["pdf", "channel", "variable", "unit", "mean", "stdev", "quant_0.01", "quant_0.25", "quant_0.5", "quant_0.75", "quant_0.9", "quant_0.99", "del_1e+07_3", "del_1e+07_10"]  # columns not typically used as filters
        filter_cols = [col for col in possible_cols if col not in static_cols]

    print("Available parameter values:\n")
    for col in filter_cols:
        unique_vals = sorted(df[col].dropna().unique())
        converted_vals = [val.item() if hasattr(val, "item") else val for val in unique_vals]
        print(f"{col}: {converted_vals}")

    filters = {}

    if user_input:
        print("Enter filtering values (single value or list like [1, 2.5]) or leave blank to skip:")
        for col in filter_cols:
            val = input(f"Filter by {col}? ")
            if val:
                try:
                    # Try to evaluate list-like input
                    val = ast.literal_eval(val)
                except:
                    pass
                filters[col] = val
    else:
        filters = filter_cols  # if user_input is False, filter_cols is assumed to be a dict

    # Apply filters
    filtered_df = df.copy()
    for col, val in filters.items():
        if isinstance(val, list):
            filtered_df = filtered_df[filtered_df[col].isin(val)]
        else:
            filtered_df = filtered_df[filtered_df[col] == val]

    # Apply all filters except "ws" for ws overview
    all_ws_filtered_df = df.copy()
    for col, val in filters.items():
        if col == "ws":
            continue  # Skip filtering by 'ws'
        if isinstance(val, list):
            all_ws_filtered_df = all_ws_filtered_df[all_ws_filtered_df[col].isin(val)]
        else:
            all_ws_filtered_df = all_ws_filtered_df[all_ws_filtered_df[col] == val]

    # Apply only "ws filter" for iec reference
    filtered_iec_df = iec_df.copy()
    if "ws" in filters:
        ws_val = filters["ws"]
        if isinstance(ws_val, list):
            filtered_iec_df = filtered_iec_df[filtered_iec_df["ws"].isin(ws_val)]
        else:
            filtered_iec_df = filtered_iec_df[filtered_iec_df["ws"] == ws_val]

    return all_ws_filtered_df, filtered_df, filtered_iec_df


#joined probability plots!


"""FOR OLD FILES
def parse_all_hist(files, wind_speeds, selected_study):
    ws_hist_dict = {}
    for ws, file in zip(wind_speeds, files):
        print(f"Parsing: {file.name}")
        df = parse_hist_file(file, selected_study)
        ws_hist_dict[ws] = df
    ws_hist_df = pd.concat(ws_hist_dict, names=["wind speed"]).reset_index()
    ws_hist_df = ws_hist_df.drop(columns=["level_1", "ws"])
    return ws_hist_df"""

"""FOR OLD FILES
def parse_all_stats(files, wind_speeds):
    ws_stats_dict = {}
    for ws, file in zip(wind_speeds, files):
        print(f"Parsing: {file.name}")
        df = parse_stats_file(file)
        df = df.filter(["channel", "mean", "stdev", "quant_0.01", "quant_0.25", "quant_0.5", "quant_0.75", "quant_0.9", "quant_0.99", "del_1e+07_3", "del_1e+07_10"])
        ws_stats_dict[ws] = df
    ws_stats_df = pd.concat(ws_stats_dict, names=["wind speed"]).reset_index()
    ws_stats_df = ws_stats_df.drop(columns="level_1")
    return ws_stats_df"""