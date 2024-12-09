import pandas as pd
import TDX_retriever as tr
import argparse
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        description="A script that gets public transportation routing result from TDX.")

    # TODO: can actually consider place the input inside another folder, eg. source_pair
    DATA_PATH = './JJinTP_data_TW/public_data/'
    API_PATH = f'{DATA_PATH}tdx_api/'

    parser.add_argument("api_file", type=str, help="Api key file name.")
    parser.add_argument("centroids", type=str, help="Input centroids file name.")
    parser.add_argument("out_file", type=str, help="Routing result file name.")
    # I think currently we don't support no-log version...
    parser.add_argument("--log", action="store_true", help="Enable logging.")
    parser.add_argument("--api_path", type=str, default=API_PATH, help=F"Api file path (default: {API_PATH}).")
    parser.add_argument("--centroid_path", type=str, default=DATA_PATH, help=f"Centroid file path (default: {DATA_PATH}).")
    parser.add_argument("--out_path", type=str, default=DATA_PATH, help=f"Output file path (default: {DATA_PATH}.")
    parser.add_argument("--depart_time", type=str, default="T10:00:00", help=f"Departure time (default: 'T10:00:00'.")
    parser.add_argument("--batch_num", type=int, default=-1, help=f"Batch number for pair files (default: -1.")
    parser.add_argument("--test", type=int, default=-1, help=f"Set the size of subset for test (default: -1, meaning not testing")


    args = parser.parse_args()

    with open(args.api_path + args.api_file) as f:
        info = json.load(f)
    app_id = info['app_id']
    app_key = info['app_key']

    # ===================================================
    #  Get pairwise from CSV
    # ===================================================
    # TODO: 
    # 2. if batch > 0, then indicating the use of batch files, or just pass in the file names??
    centroids = pd.read_csv(args.centroid_path + args.centroids)

    TDX = tr.TDX_retriever(app_id, app_key, add_villcode=True)

    # -------------
    #  For testing
    # -------------
    test_size = args.test

    if test_size > 0:
        centroids = centroids.iloc[:test_size]

    # ----------
    #  keywords
    # ----------
    batch_num = args.batch_num
    time_format = args.depart_time
    cur_hour = int(time_format[1:3])
    ampm = 'am' if cur_hour < 12 or cur_hour == 24 else 'pm'
    hour = 0 if cur_hour == 24 else (cur_hour % 12 or 12)
    time_symb = f"{hour}{ampm}"

    print(f"Start Processing Centroids with depart time at {time_symb}...")

    dt_dtr = datetime.now().strftime("%Y%m%d")
    bname = '' if batch_num < 0 else f'_{batch_num}'
    TDX.set_log_name(
        log_name=f"{dt_dtr}_{time_symb}{bname}.log"
    )

    TDX._set_condition(target_time=time_format)
    df = TDX.get_pairwise_paired(centroids)

    df.to_csv(args.out_path + args.out_file, index=False)


if __name__ == "__main__":
    main()
