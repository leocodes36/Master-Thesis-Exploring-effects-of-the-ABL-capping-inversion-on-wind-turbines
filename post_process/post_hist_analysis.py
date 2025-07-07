from post_common import *
from post_plotting import *
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 18,
    "font.size": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

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
    
    iec_hist_files = list_iec_hist_files(CONFIG["data_path"])
    iec_hist_df = parse_iec_hist(iec_hist_files)

    hist_files = list_hist_files(selected_study, CONFIG["data_path"])
    hist_df = parse_all_hist(hist_files, selected_study)

    print("\nMerged data preview:")
    pd.set_option('display.max_columns', None)
    print(hist_df.head())

    #plot_selected_blade_pdfs_with_sliders(hist_df, iec_hist_df)
    #plot_channel_pdf_with_sliders(hist_df, iec_hist_df, 17)
    
    all_ws_filtered_df, filtered_df, filtered_iec_df = filter_dataframe(hist_df, iec_hist_df)
    
    print("\nPlotting tower and blades by default")
    plot_selected_channel_pdfs(filtered_df, filtered_iec_df)
    print("\nChannels available:", CONFIG["default_channels"])
    ch_choice = input("\nEnter the channel you want to assess: ")
    if ch_choice != "":
        ch_choice = int(ch_choice)
        plot_other_channel(filtered_df, filtered_iec_df, ch_choice)
    
    plt.show()


if __name__ == "__main__":
    main()