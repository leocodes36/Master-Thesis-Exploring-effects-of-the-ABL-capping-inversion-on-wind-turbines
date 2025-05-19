from post_common import *
from post_plotting import *

CONFIG = {
    "data_path": Path("/Users/leo/python/thesis/sync_results/"),
    "lookup_csv": Path("/Users/leo/python/thesis/post_process/HAWC2_Output_Channels.csv"),
    "default_channels": [10, 12, 15, 17, 26, 29, 32],
    "fatigue_columns": {0: "del_1e+07_3", 1: "del_1e+07_10"},
}

def main():
    args = parse_arguments()
    
    study_names = load_study_names(CONFIG["data_path"])

    if args.interactive or not args.study:
        selected_study = select_study(study_names)
    else:
        if args.study not in study_names:
            raise ValueError(f"Study '{args.study}' not found.")
        selected_study = args.study

    iec_stats_files = list_iec_stats_files(CONFIG["data_path"])
    iec_stats_df = parse_iec_stats(iec_stats_files)

    stats_files = list_stats_files(selected_study, CONFIG["data_path"])
    stats_df = parse_all_stats(stats_files)

    lookupdf = pd.read_csv(CONFIG["lookup_csv"])
    lookupdf.columns = lookupdf.columns.str.strip().str.lower()
    lookupdf = lookupdf[["channel", "variable", "unit"]]
    stats_df = stats_df.merge(lookupdf, on="channel", how="left")

    print("\nMerged data preview:")
    pd.set_option('display.max_columns', None)
    print(stats_df.head())

    all_ws_filtered_df, filtered_df, filtered_iec_df = filter_dataframe(stats_df, iec_stats_df)
    
    plot_channelstats(filtered_df, filtered_iec_df)
    print("\nDefault channels available:", CONFIG["default_channels"])
    ch_choice = int(input("\nEnter the channel you want to assess: "))
    print(f"\nPlotting Statistics for Channel {ch_choice}")
    plot_channel_ws_overview(all_ws_filtered_df, iec_stats_df, ch_choice)

    if selected_study == "cnbl":
        choice = input("\nWant to see awesome joined probability plots? [y/n]")
        if choice.strip() == "y":
            s_choice = int(input("\nEnter the jet strength you want to assess: "))
            plot_joint_probs(stats_df, s_choice)
            plot_joint_probs2(stats_df)

    plt.show()


if __name__ == "__main__":
    main()
