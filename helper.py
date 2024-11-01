import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from collections import defaultdict
import numpy as np
import json
import os
import ast

OUT_PATH = "output/"
DATA_PATH = "JJinTP_data_TW/"
LOG_PATH = "log/"


class Helper():
    def __init__(self, destination: str = "./JJinTP_data_TW/Routing/") -> None:
        self.path = destination

    def get_in_pair_slow(self, infile: str = "village_centroid_TP.csv", outfile: str = "in_pairs.csv"):
        '''
        Using nested for loop is extremely slow.
        I never finished running this code cuz I'm impatient.
        But I guess it would take around 1.5 hours?
        '''
        centroids = pd.read_csv(self.path+infile)

        n = centroids.shape[0]

        df = pd.DataFrame(
            columns=['A_VCODE', 'A_lon', 'A_lat', 'B_VCODE', 'B_lon', 'B_lat']
        )

        with tqdm(total=n * (n-1) // 2) as pbar:
            for i in range(n):
                for j in range(i+1, n):
                    df.loc[pbar.n] = [
                        centroids['VILLCODE'].iloc[i],
                        centroids['lon'].iloc[i],
                        centroids['lat'].iloc[i],
                        centroids['VILLCODE'].iloc[j],
                        centroids['lon'].iloc[j],
                        centroids['lat'].iloc[j]
                    ]
                    pbar.update(1)

        df[['A_VCODE', 'B_VCODE']] = df[['A_VCODE', 'B_VCODE']].astype(int)

        df.to_csv(self.path+outfile, index=False)
        return

    def get_in_pair(self, infile: str = "village_centroid_TP.csv", outfile: str = "in_pairs.csv"):
        '''
        Make the village centers into pairs from a list of center points.

        Using matrix manipulation is way faster than I thought.
        Excluding the file ouputting time, this code finished within about 15 sec.
        '''
        centroids = pd.read_csv(self.path+infile)

        n = centroids.shape[0]

        tt = np.array(centroids)  # 2d array
        tt = tt.tolist()  # list of lists

        with tqdm(total=(n * (n-1) // 2), desc="Working on the large matrix") as pbar:
            result = [
                # the '+' here is list operation
                # turn the list into string to do matrix things
                [str(x + y) for y in tt]
                for x in tt
            ]

            for x in tt:
                for _ in tt:
                    pbar.update(1)

        result = np.array(result)

        # this is a list of size 850860, with elements looks like:
        # [63000080031.0, 121.55598430669878, 24.9788580602204, 63000080032.0, 121.56397728822247, 24.98154918033328]
        above_diagonal = result[np.triu_indices_from(result, k=1)]

        df = pd.DataFrame(
            # literal_eval is a safer version of eval on string
            data=list(map(literal_eval, above_diagonal)),
            columns=['A_VCODE', 'A_lon', 'A_lat', 'B_VCODE', 'B_lon', 'B_lat']
        )

        df[['A_VCODE', 'B_VCODE']] = df[['A_VCODE', 'B_VCODE']].astype(int)

        df.to_csv(self.path+outfile, index=False)
        return

    def data_into_x_splits(self, infile: str = "in_pairs.csv", x: int = 3):
        '''
        Split the paired data into sub files, since the TDX server does not support
        accessing more than 200,000 times a day, and there are about 800,000 rows 
        in our paired list, each row will trigger 2 times of access to the server.
        '''
        df = pd.read_csv(self.path+infile)
        n = df.shape[0]
        gap = n // x + 1
        cur_head = 0

        for i in range(x):
            tmp_df = df.iloc[cur_head:cur_head+gap]

            tmp_df.to_csv(f"{self.path}in_pairs_sub{i+1}.csv", index=False)
            print(f"Sub{i+1} created.")

            cur_head += gap
        return

    def task_generator(self, file_list: list = [1], keys: int = 1):
        '''
        This method generates the task file in vscode, so we don't 
        need to manually open several terminals and type the commands.
        '''
        sub_task_cnt = len(file_list)
        batch_size = sub_task_cnt // keys

        # Actually, here we only need to set the vscode interpretor to
        # the env, then can run without a line of script to activate.
        # activate_venv = "env/Scripts/Activate.ps1" # for powershell

        tasks = [
            {
                "label": f"get TDX with key {n // batch_size + 1} on file {file_list[n]}",
                "type": "shell",
                "command": f"env\\run_py_key{n // batch_size + 1}.bat {file_list[n]}"
            }
            for n in range(sub_task_cnt)
        ]

        taskall = {
            "label": "run all",
            "dependsOn": [t['label'] for t in tasks],
            "dependsOrder": "parallel",
            "presentation": {
                "reveal": "always",
                "revealProblems": "onProblem",
                "panel": "new"
            }
        }

        tasks.append(taskall)

        to_json = {
            "version": "2.0.0",
            "tasks": tasks
        }

        with open(f'{self.path}tasks.json', 'w') as fp:
            json.dump(
                to_json, fp,
                sort_keys=False, indent=4, separators=(',', ': ')
            )

    # ABOVE FUNCTIONS ARE MAINLY FOR GETTING DATA FROM TDX
    ######################################################

    def merge_public_files(self, start_time: str = "6pm", file_cnt: int = 20):
        '''
        This function will merge the series of files from 1 to 'file_cnt'
        with the stated 'start_time'

        columns: A_villcode,B_villcode,A_lat,A_lon,B_lat,B_lon,AB_travel_time,
                 AB_ttl_cost,AB_transfer_cnt,AB_route,BA_travel_time,BA_ttl_cost,
                 BA_transfer_cnt,BA_route

        Parameters
        ----------
        start_time: str
            This is the departure time for public transportation, 
            only 6pm and 10am available.
        file_cnt: int
            The number of files needed to be merged.
        '''
        out_file = f"{OUT_PATH}merged_public_{start_time}.csv"

        if os.path.exists(out_file):
            print(
                f'The file already exists in {OUT_PATH}, skipping this step...')
            return

        file_paths = [
            f"{OUT_PATH}travel_at_{start_time}_{s}.csv" for s in range(1, file_cnt+1)]

        # Load, concatenate, and save as a single file
        df = pd.concat([pd.read_csv(f) for f in file_paths])
        df.to_csv(out_file, index=False)
        return

    def __flawed_village_to_township(self, mode: str, calib_info: pd.DataFrame, to_file: bool = True):
        '''
        This function is to merge the village-wise travel times to township-wise
        public columns: A_villcode,B_villcode,A_lat,A_lon,B_lat,B_lon,AB_travel_time,AB_ttl_cost,AB_transfer_cnt,AB_route,BA_travel_time,BA_ttl_cost,BA_transfer_cnt,BA_route
        private columns: id_orig,lon_orig,lat_orig,id_dest,lon_dest,lat_dest,query_status,timeofday,dayofweek,date,weight_name,distance,duration

        Parameters
        ----------
        mode: str.
            only supports 'private' and 'public'

        Return
        ------
            the township-wise travel time
        '''
        mode = mode.lower()
        if mode == 'public':
            # I manually copy the merged data to this directory and renamed (remove the timestamp)
            # note that in this data public transport unavailble is NaN, but later pivot_table fill 0 will automatically turn them into 0's
            df = pd.read_csv(f"{DATA_PATH}public_data/merged_public.csv")
            a_col = 'A_villcode'
            b_col = 'B_villcode'
            ab_time_col = 'AB_travel_time'
            ba_time_col = 'BA_travel_time'
        elif mode == 'private':
            df = pd.read_csv(
                f"{DATA_PATH}car_data/village_centroid_TP_output_20240605143517.csv")
            a_col = 'id_orig'
            b_col = 'id_dest'
            # back and forth share the same time for driving
            ab_time_col = 'duration'
            ba_time_col = 'duration'
        else:
            raise ValueError("receive only either 'public' or 'private'.")
        print(f"Currently working on {mode}!")

        # Filter the villages
        keep_villcodes = calib_info['VILLCODE']
        df = df[df[a_col].isin(keep_villcodes)]
        df = df[df[b_col].isin(keep_villcodes)]

        all_points = sorted(set(df[a_col]).union(df[b_col]))

        # ---------------------------------------------
        #  Turn our travel table back into pair matrix
        # ---------------------------------------------
        def to_mat(df: pd.DataFrame):
            # the following part could be clearer if you run t_back_to_mat() in pg.py
            ab_matrix = df.pivot_table(
                index=a_col, columns=b_col, values=ab_time_col, fill_value=0)
            ba_matrix = df.pivot_table(
                index=b_col, columns=a_col, values=ba_time_col, fill_value=0)

            # If the villcode were sorted when generating the data,
            # ab_matrix should be the upper triangle and ba_matrix should be the lower
            ab_matrix = ab_matrix.reindex(
                index=all_points, columns=all_points, fill_value=0)
            ba_matrix = ba_matrix.reindex(
                index=all_points, columns=all_points, fill_value=0)

            # !! BUT !! not sorted is still OK, as long as the diagonal is all 0
            mat = ab_matrix + ba_matrix

            # check diagonal
            is_diagonal_zero = np.diag(mat.values).all() == 0
            print("All diagonal values are 0:", is_diagonal_zero)
            print("Back to matrix done!")

            return mat

        # mat = to_mat(df)

        # ----------------------------------------------
        #  Recording villcode pairs with no travel time
        # ----------------------------------------------
        def recording_pairs_no_travel_time(travel_mat):
            zero_indices = np.argwhere(travel_mat == 0)
            # Filter out diagonal elements, here x and y are indices like 0,1,2...
            non_diagonal_zeros = defaultdict(list)
            for x, y in zero_indices:
                if x != y:
                    non_diagonal_zeros[x].append(y)

            # save as the index corresponding villcodes
            non_diagonal_zeros = {
                all_points[k]: [all_points[v] for v in values]
                for k, values in non_diagonal_zeros.items()
            }
            with open(f'{LOG_PATH}zeros_{mode}.json', 'w') as f:
                json.dump(non_diagonal_zeros, f, indent=4,
                          separators=(',', ': '))

        # recording_pairs_no_travel_time(mat)

        # ===================================
        #  Start merging village to township
        # ===================================
        # NOTE: since computing Township level using the original table would
        #       be easier, we don't need matrix now
        # 1. get the weights for each villcode by population
        calib_info['township'] = calib_info['VILLCODE'].astype("str").str[:-3]
        calib_info['weight'] = calib_info['population'] / \
            calib_info.groupby('township')['population'].transform('sum')
        weights = calib_info[['VILLCODE', 'weight']]

        # 2. A and B cols make their township code: township_A and township_B
        df['A_township'] = df[a_col].astype("str").str[:-3]
        df['B_township'] = df[b_col].astype("str").str[:-3]

        start_pt = {
            'A': {'dest': 'B', 'vill_a': a_col, 'vill_b': b_col, 'time': ab_time_col},
            'B': {'dest': 'A', 'vill_a': b_col, 'vill_b': a_col, 'time': ba_time_col}
        }

        township_travel_dfs = []

        # following steps will use A to B as example
        for start, info in start_pt.items():
            # 3. AB_time x weights mapped with A
            merged_df = df.merge(
                weights.rename(columns={'weight': 'weight_A'}),
                left_on=info['vill_a'], right_on='VILLCODE', how='left'
            )
            merged_df['weighted_time_A'] = merged_df[info['time']] * \
                merged_df['weight_A']

            # 4. for each A_township (A or B doesn't matter, should have the same amount),
            #       group by B: sum the products from prev step
            #       and then times with the weights mapped with B
            #       and then group sum by township_B
            for town in set(df[f'{start}_township']):
                town_df = merged_df[merged_df[f'{start}_township'] == town]
                # 4-1
                by_villb_df = town_df.groupby(
                    [f"{info['dest']}_township", info['vill_b']], as_index=False
                )['weighted_time_A'].sum()
                # 4-2
                by_villb_df = by_villb_df.merge(
                    weights.rename(columns={'weight': 'weight_B'}),
                    left_on=info['vill_b'], right_on='VILLCODE', how='left'
                )
                by_villb_df['weighted_time_B'] = by_villb_df['weighted_time_A'] * \
                    by_villb_df['weight_B']
                # 4-3
                result = by_villb_df.groupby(
                    f"{info['dest']}_township", as_index=False
                )['weighted_time_B'].sum()

                # Force all the dest column to be B_township (even if it was actually A)
                # since the layout of this travel data would not be the same as in village
                # this will be just A, B, time for all pairs (was A, B, AB_time, BA_time)
                result.columns = ['B_township', 'time']
                result['A_township'] = town
                # rearrange column order
                result = result[['A_township', 'B_township', 'time']]

                township_travel_dfs.append(result)

        town_travel_times = pd.concat(
            township_travel_dfs, ignore_index=True)  # 1178

        # Check for duplicate rows based on 'col1' and 'col2'
        duplicates = town_travel_times[town_travel_times.duplicated(
            subset=['A_township', 'B_township'], keep=False)]

        # print(duplicates)

        return
        if to_file:
            df.to_csv()
            return
        return df

    def village_to_township(self, mode: str, calib_info: pd.DataFrame, to_file: bool = True):
        '''
        This function is to merge the village-wise travel times to township-wise
        public columns: A_villcode,B_villcode,A_lat,A_lon,B_lat,B_lon,AB_travel_time,AB_ttl_cost,AB_transfer_cnt,AB_route,BA_travel_time,BA_ttl_cost,BA_transfer_cnt,BA_route
        private columns: id_orig,lon_orig,lat_orig,id_dest,lon_dest,lat_dest,query_status,timeofday,dayofweek,date,weight_name,distance,duration

        Parameters
        ----------
        mode: str.
            only supports 'private' and 'public'

        Return
        ------
            the township-wise travel time
        '''
        mode = mode.lower()
        if mode == 'public':
            # I manually copy the merged data to this directory and renamed (remove the timestamp)
            # note that in this data public transport unavailble is NaN, but later pivot_table fill 0 will automatically turn them into 0's
            folded_pair = pd.read_csv(
                f"{DATA_PATH}public_data/merged_public.csv")
            a_col = 'A_villcode'
            b_col = 'B_villcode'
            ab_time_col = 'AB_travel_time'
            ba_time_col = 'BA_travel_time'
        elif mode == 'private':
            folded_pair = pd.read_csv(
                f"{DATA_PATH}car_data/village_centroid_TP_output_20240605143517.csv")
            a_col = 'id_orig'
            b_col = 'id_dest'
            # back and forth share the same time for driving
            ab_time_col = 'duration'
            ba_time_col = 'duration'
        else:
            raise ValueError("receive only either 'public' or 'private'.")
        print(f"Currently working on {mode}!")

        # Filter the villages
        keep_villcodes = calib_info['VILLCODE']
        folded_pair = folded_pair[folded_pair[a_col].isin(keep_villcodes)]
        folded_pair = folded_pair[folded_pair[b_col].isin(keep_villcodes)]

        # ===================================
        #  Start merging village to township
        # ===================================
        # NOTE: since computing Township level using the original table would
        #       be easier, we don't need matrix now
        # 1. get the weights for each villcode by population
        calib_info['township'] = calib_info['VILLCODE'].astype("str").str[:-3]
        calib_info['weight'] = calib_info['population'] / \
            calib_info.groupby('township')['population'].transform('sum')
        weights = calib_info[['VILLCODE', 'weight']]

        # 2. Concate A to B and B to A into a long data with columns:
        #    vill_a, vill_b, time
        new_cols = ['vill_a', 'vill_b', 'time']
        AB_df = folded_pair[[a_col, b_col, ab_time_col]]
        AB_df.columns = new_cols
        BA_df = folded_pair[[b_col, a_col, ba_time_col]]
        BA_df.columns = new_cols

        all_pair = pd.concat([AB_df, BA_df], ignore_index=True)

        all_pair['town_a'] = all_pair['vill_a'].astype("str").str[:-3]
        all_pair['town_b'] = all_pair['vill_b'].astype("str").str[:-3]

        # each element is the travel times from single town to all other towns
        # (including itself)
        township_travel_dfs = []

        # 3. AB_time x weights mapped to A
        merged_df = all_pair.merge(
            weights.rename(columns={'weight': 'weight_A'}),
            left_on='vill_a', right_on='VILLCODE', how='left'
        )
        merged_df['weighted_t_A'] = merged_df['time'] * merged_df['weight_A']

        # 4. for each town_A (A or B doesn't matter, should have the same amount),
        #       group by vill_B: sum the products from prev step
        #       and then times with the weights mapped with vill_B
        #       and then group sum by town_B
        for town in set(all_pair['town_a']):
            town_df = merged_df[merged_df['town_a'] == town]
            # 4-1
            by_villb_df = town_df.groupby(
                ['town_b', 'vill_b'], as_index=False)['weighted_t_A'].sum()
            # 4-2
            by_villb_df = by_villb_df.merge(
                weights.rename(columns={'weight': 'weight_B'}),
                left_on='vill_b', right_on='VILLCODE', how='left'
            )
            by_villb_df['weighted_t_B'] = by_villb_df['weighted_t_A'] * \
                by_villb_df['weight_B']
            # 4-3
            result = by_villb_df.groupby(
                'town_b', as_index=False)['weighted_t_B'].sum()

            # Force all the dest column to be B_township (even if it was actually A)
            # since the layout of this travel data would not be the same as in village
            # this will be just A, B, time for all pairs (was A, B, AB_time, BA_time)
            result.columns = ['town_b', 'time']
            result['town_a'] = town
            # rearrange column order
            result = result[['town_a', 'town_b', 'time']]

            township_travel_dfs.append(result)

        town_travel_times = pd.concat(
            township_travel_dfs, ignore_index=True)  # 961

        town_time_mat = town_travel_times.pivot_table(
            index='town_a', columns='town_b', values='time', fill_value=0)

        # 如果要ordered index就參考下面的
        # ab_matrix = ab_matrix.reindex(
        #         index=all_points, columns=all_points, fill_value=0)

        # TODO: now what? use matrix to do things?

        return
        if to_file:
            df.to_csv()
            return
        return df

    def process_survey(self, years: list):
        '''
        This function will read all the raw survey data, clean them, and then
        save the cleaned and merged data to a new file.
        '''
        # From codebook
        # 1. township code at A192:B585 (NTP: 1XX, TP: 22XX)
        town_code = pd.read_csv(f"{DATA_PATH}survey_data/變數名稱說明檔_98-105.csv")
        town_code = town_code.iloc[191:, :2]
        town_code.columns = ['code', 'town']

        # drop if code ends with 00, meaning the city. eg. 100  @@新北市(臺北縣)
        town_code = town_code[~town_code['code'].str.endswith('00')]
        # keep code in range of New Taipei: 100~199 and Taipei: 2200~2299
        town_code['code'] = town_code['code'].astype(int)  # Convert to int
        cond_ntp = town_code['code'].between(100, 199)
        cond_tp = town_code['code'].between(2200, 2299)
        town_code = town_code[cond_ntp | cond_tp]

        # Base the townships needed on the calibration data
        town_code_TP = pd.read_csv(f"{DATA_PATH}calibration_data_TP.csv")
        town_code_TP['TOWNCODE'] = town_code_TP['VILLCODE']\
            .astype("str").str[:-3]
        town_code_TP = town_code_TP[['TOWNCODE', 'TOWNNAME']]
        town_code_TP = town_code_TP.drop_duplicates(
            subset='TOWNCODE', keep='first').reset_index(drop=True)
        town_code_TP = town_code_TP.merge(
            town_code, left_on='TOWNNAME', right_on='town', how='left')

        # rearrange columns
        # NOTE: 'TOWNCODE' is for travel cost, 'code' is for survey data
        town_code_TP = town_code_TP[['TOWNCODE', 'code', 'TOWNNAME']]

        # 2. transport mode D192:E217
        #    1 捷運, 2 市區公車, 3 公路客運, 4 計程車(含共乘), 5 臺鐵, 6 高鐵, 7 渡輪, 8 交通車, 9 免費公車,
        #    10 國道客運, 11 飛機, 12 步行, 13 自行車, 14 機車, 15 自用小客車(含小客貨兩用車), 16 自用大客車,
        #    17 自用小貨車, 18 自用大貨車(含大客貨兩用車), 19 免費公車, 31 國道客運, 32 公路客運, 33 復康巴士,
        #    97 其他, 98 不知道/拒答, NULL 未填答
        # this file is manual extraction from the 變數名稱說明檔_98-105.csv
        mode_codes = pd.read_csv(
            f"{DATA_PATH}survey_data/transportation_code.csv")

        mode_codes['code'] = mode_codes['code'].astype(int)

        fine_mode = mode_codes[['code', 'mode']]
        coarse_mode = mode_codes[['code', 'coarse_mode']]
        public_private = mode_codes[['code', 'public_private']]

        # ================
        #  Start Cleaning
        # ================
        # only keep the mapping of columns we need.
        col_name_map = {
            'v13': 'Age',
            'v12': 'Gender',  # 1: men, 2: women
            'v2_1': 'Residence',  # = 'code', check "town_code_TP" above
            'v3_1': 'Working_or_studying',  # 1: work, 2: study, 3: both, 98: none
            'v4A': 'Workplace',  # check "town_code_TP"
            'v4B': 'School',  # check "town_code_TP"
            'v5_1': 'Transportation_1',  # mode_codes for all columns below
            'v5_2': 'Transportation_2',
            'v5_3': 'Transportation_3',
            'v5_4': 'Transportation_4',
            'v5_5': 'Transportation_5',
            'v5_6': 'Transportation_6',
            'v5_7': 'Transportation_7',
            'v5_8': 'Transportation_8',
            'v5_9': 'Transportation_9',
            'v5_10': 'Transportation_10',  # 較常使用
            'v6': 'Daily_transport'  # 最常使用
        }

        survey_files = []

        for y in years:
            file_path = f"{DATA_PATH}survey_data/{y}年民眾日常使用運具狀況調查原始資料.csv"
            cur_file = pd.read_csv(file_path)

            # 1. for each file, keep only the columns above, make a year column
            #    and then stack them up
            cur_file = cur_file[col_name_map.keys()]
            cur_file.rename(columns=col_name_map, inplace=True)

            cur_file['year'] = y + 1911

            survey_files.append(cur_file)

        survey_df = pd.concat(survey_files, ignore_index=True)
        # but here we can replace empty strings with 0's is because there are no 0's in this dataset
        survey_df = survey_df.astype(str).map(lambda x: x.strip())\
            .replace('', '0').astype(int)

        # check na or 0 in each column
        check = survey_df.apply(lambda x: x.isna() | (x == 0))
        check = check.sum()
        print(f"NaN and 0 (empty string) check:\n{check}")

        # 2. get the town code used in travel cost "TOWNCODE"
        # right join to keep only the filtered districts (since town_code_TP was filtered above)
        survey_df = survey_df.merge(
            town_code_TP, left_on='Residence', right_on='code', how='right'
        )

        # TODO: also filter the destination?
        # 需要跟老師確認，住範圍但通勤到範圍外的要留嗎
        # 另外也要確認我們最終要用的是以居住地來看的mode percentage
        # 還是居住-通勤pair的 mode percentage (這個每個pair筆數一定會很少)

        # 3. count the occurance of mode/coarse_mode/public_private across years
        #    all sample
        #    group by TOWNCODE
        survey_df['Daily_transport'] = survey_df['Daily_transport'].fillna(98)

        survey_df = survey_df.merge(
            mode_codes, left_on='Daily_transport', right_on='code', how='left'
        )
        modes_to_check = ['mode', 'coarse_mode', 'public_private']
        for current_mode in modes_to_check:
            print(f"Current mode is: {current_mode}")
            # 3-1
            counts = survey_df[current_mode]\
                .value_counts().reset_index(name='cnt')
            counts['pct'] = (counts['cnt'] / counts['cnt'].sum()) * 100
            counts['pct'] = counts['pct'].round(1)

            print(counts)
            # print(sum(counts['cnt']), sum(counts['pct']))
            # 3-2
            counts = survey_df.groupby('TOWNCODE')[current_mode]\
                .value_counts().reset_index(name='cnt')
            counts['pct'] = (counts['cnt'] / counts.groupby('TOWNCODE')['cnt']
                            .transform('sum')) * 100
            counts['pct'] = counts['pct'].round(1)
            # print(counts)

            # (opt.) group by TOWNCODE, count the occurance of codes in Transportation_1-10
            # 但這個方法就是很難定義分母 (不可能是sample size，也許是用all non-na count)


def TDX_helper():
    h = Helper()
    # h.get_in_pair() # this generates the "in_pair.csv"

    remake_sub = input("Do you wish to remake subfiles?(Y/N)")
    if remake_sub.lower() == 'y':
        sub_file_cnt = input("Sub files to generate: ")
        if sub_file_cnt.isnumeric():
            sub_file_cnt = int(sub_file_cnt)
        else:
            raise ValueError("Please provide a valid value.")
        # this splits in_pair.csv into sub files for multi tasking.
        h.data_into_x_splits(x=sub_file_cnt)

    remake_task = input("Do you wish to remake task?(Y/N)")
    if remake_task.lower() == 'y':
        start = int(input("Serial number of file to start: "))
        n = int(input("Number of files wish to process: "))
        file_list = list(range(start, start+n))
        h2 = Helper("./.vscode/")
        h2.task_generator(file_list=file_list, keys=1)
    else:
        print("Program ending...")


def travel_cost_helper():
    h = Helper()
    # =============================
    #  Merge Public Transport Data
    # =============================
    # h.merge_public_files(calib['VILLCODE']) # this generates merged_public_{time}.csv

    # ============================================================
    #  Combine the Village Level Transportation to Township Level
    # ============================================================
    # calibration columns: VILLCODE,COUNTYNAME,TOWNNAME,VILLNAME,area,num_est,employment,avg_wage,num_est_ls,employment_ls,total_revenue_ls,total_value_added_ls,avg_employment_ls,population,floorspace,floorspace_R,floorspace_C,avg_price_R,avg_price_C,pop_den,employ_den
    calib = pd.read_csv(f"{DATA_PATH}calibration_data_TP.csv")  # 1247
    calib = calib[['VILLCODE', 'area', 'employment', 'population']]

    # 先用private做測試
    # h.village_to_township(mode='public', calib_info=calib)
    h.village_to_township(mode='private', calib_info=calib)


def main():
    # =============
    #  Survey Data
    # =============
    h = Helper()
    years = list(range(98, 106))  # ROC 98 ~ 105
    h.process_survey(years=years)

    # =============
    #  Travel Cost
    # =============
    # travel_cost_helper()


if __name__ == "__main__":
    main()
