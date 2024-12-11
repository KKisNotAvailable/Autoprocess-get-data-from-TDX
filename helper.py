import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from collections import defaultdict, Counter
import numpy as np
import json
import os
import ast

OUT_PATH = "output/"
DATA_PATH = "JJinTP_data_TW/"
LOG_PATH = "log/"

FILE_CALIB = "JJinTP_data_TW/calibration_data_TP.csv"
FILE_VILL_CENTROID = "JJinTP_data_TW/Routing/village_centroid_TP.csv"


class Helper_tdx():
    # TODO: need to change where the batch files destination path.
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

    def data_into_x_splits(self, x: int, file_path: str, infile: str):
        '''
        Split the paired data into sub files, since the TDX server does not support
        accessing more than 200,000 times a day, and there are about 800,000 rows 
        in our paired list, each row will trigger 2 times of access to the server.
        '''
        df = pd.read_csv(file_path+infile)
        n = df.shape[0]
        gap = n // x + 1
        cur_head = 0

        for i in range(x):
            tmp_df = df.iloc[cur_head:cur_head+gap]

            tmp_df.to_csv(
                f"{file_path}{infile.replace('.csv', '')}_sub{i+1}.csv", index=False)
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


class Helper_public_travel():
    def __init__(self, out_path=OUT_PATH, calib_path='', centroid_path='') -> None:
        self.__out_path = out_path

        calib = pd.read_csv(calib_path)  # 1247
        calib = calib[['VILLCODE', 'area', 'employment',
                       'population', 'TOWNNAME', 'VILLNAME']]
        self.__calib = calib

        # VILLCODE,lon,lat
        centroids = pd.read_csv(centroid_path)
        self.__centroids = centroids

    def _public_data_check(self, fpath):
        '''
        Mainly generates to files for centroid validity check: 
        1. public_problem_pairs: for checking and later extract pairs to get public 
                                 travel time again
        2. problem_vills_list: renamed to problem_vills after I've done the centroid review

        Parameters
        ----------
        fpath: str.
            the public data file path.
        calib_info: pd.DataFrame.
            the calibration data contains filtered list of villages, we need the 
            VILLCODE and the village's chinese name for manual checking

        Return
        ------
            None
        '''
        calib_info = self.__calib

        # since both set and list are not hashable, turn the sorted list into string
        data = pd.read_csv(fpath)
        data = data[['A_villcode', 'B_villcode', 'AB_travel_time', 'BA_travel_time']]
        data['point_set'] = [",".join(map(str, sorted(pair))) for pair in zip(data['A_villcode'], data['B_villcode'])]

        walk_data = pd.read_csv(f"{DATA_PATH}public_data/travel_walking.csv")
        walk_data['point_set'] = [",".join(map(str, sorted(pair))) for pair in zip(
            walk_data['id_orig'], walk_data['id_dest'])]
        walk_data = walk_data[['point_set', 'duration']]

        calib_info['name_ch'] = [t + v for t,
                                 v in zip(calib_info['TOWNNAME'], calib_info['VILLNAME'])]
        calib_info = calib_info[['VILLCODE', 'name_ch']]

        # print(data.head(5))

        # 1. Preserve the valid VILLCODE (could be ignored when merging village to township, there's a step doing this)
        keep_villcodes = calib_info['VILLCODE']
        data = data[data['A_villcode'].isin(keep_villcodes)]
        data = data[data['B_villcode'].isin(keep_villcodes)]

        # get the chinese names
        data = data.merge(
            calib_info.rename(
                columns={'VILLCODE': 'A_villcode', 'name_ch': 'name_A'}),
            on='A_villcode', how='left'
        )
        data = data.merge(
            calib_info.rename(
                columns={'VILLCODE': 'B_villcode', 'name_ch': 'name_B'}),
            on='B_villcode', how='left'
        )

        # 2. merge with walking data, filter out the walk time <= 30 min (1800 sec)
        data = data.merge(walk_data, on='point_set', how='left')
        data = data[data['duration'] > 1800]

        # check AB_travel_time and BA_travel_time still empty
        check = data[(data['AB_travel_time'].isna()) |
                     (data['BA_travel_time'].isna())]
        check = check[[
            'A_villcode', 'B_villcode', 'AB_travel_time', 'BA_travel_time',
            'name_A', 'name_B', 'duration'
        ]].rename(columns={'duration': 'walking_time'})

        # print(check)
        # check.to_csv(OUT_PATH+"public_problem_pairs_rerun1.csv", index=False)

        # Get the count of the village occurance and output as file for
        # manual recording
        code_cnt = Counter(
            list(check['A_villcode']) + list(check['B_villcode']))
        vill_cnt = Counter(list(check['name_A']) + list(check['name_B']))
        v_col, c_col, cnt_col = [], [], []
        for k, v in vill_cnt.most_common():
            v_col.append(k)
            cnt_col.append(v)
        for k, v in code_cnt.most_common():
            c_col.append(k)

        problem_vills = pd.DataFrame({
            'VILLCODE': c_col,
            'NAME': v_col,
            'CNT': cnt_col
        })

        problem_vills['OPERATION'] = ""
        problem_vills['NOTE'] = ""
        # problem_vills was based on problem_vills_list. I manully checked
        # the centroids on map and record the operations.
        # The renewed cnetroids are in another file.
        # problem_vills.to_csv(OUT_PATH+"problem_vills_list.csv", index=False)

    def _check_centroid_changed_cnt(self):
        '''
        After manully checking with google maps, centroids of some of 
        the villages were re-assigned. Thus, this function is to check
        if the recorded changed villages is the same amount as the centroid.
        '''
        # from the problem.vill.csv => get '重訂座標'
        record_df = pd.read_csv(OUT_PATH+'problem_vills.csv')
        print(sum(record_df['OPERATION'] == '重訂座標'))

        # from village_centroid_TP.csv => get the shorter lon lat
        centroids_df = pd.read_csv(
            f'{DATA_PATH}Routing/village_centroid_TP.csv', dtype={'lon': 'str'})
        print(sum(centroids_df['lon'].map(len) < 12))

        # Both are 57

    def merge_public_files(
        self, source_path, out_path="",
        start_time="6pm", file_cnt=20
    ):
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
        if not out_path:
            out_path = self.__out_path

        out_file = f"{out_path}merged_public_{start_time}.csv"

        if os.path.exists(out_file):
            print(
                f'The file already exists in {out_path}, skipping this step...')
            return

        file_paths = [
            f"{source_path}travel_at_{start_time}_{s}.csv" for s in range(1, file_cnt+1)]

        # Load, concatenate, and save as a single file
        df = pd.concat([pd.read_csv(f) for f in file_paths])
        df.to_csv(out_file, index=False)
        return

    def get_recentered_list(self, problem_vills_fpath, out_fpath):
        '''
        In the problem_vills file, OPERATION is used to recognize the 
        recentered villages. This function will directly generates file
        as the out_fpath.

        Parameters
        ----------
        problem_vills_file: str.
            the file path + name to locate the problem_vills.csv

        Return
        ------
            None.
        '''
        data = pd.read_csv(problem_vills_fpath)

        data = data[data['OPERATION'] == '重訂座標'].reset_index(drop=True)
        data['VILLCODE'].to_csv(out_fpath, index=False)

    def get_rerun_pairs(
            self, problem_vills_fpath, problem_pairs_fpath, recentered_fpath, 
            out_fpath, include_recenter_pairs_in_rerun=True, 
            exclude_recenter_pairs_in_probpair=True
        ):
        '''
        This function generates the pairs for TDX to rerun from the 
        public_problem_pairs.csv

        Parameters
        ----------
        problem_vills_fpath: str.
            the file of villages with records if they have been recentered.
        problem_pairs_fpath: str.
            the file of problem paris.
        recentered_fpath: str.
            the file to locate the recentered_list.csv, generated from problem_vills_fpath
        include_recenter_pairs_in_rerun: bool.
            to decide whether to include the pairs including recentered 
            points. If False then only the non-recentered pairs in the 
            public_problem_pairs.csv would be rerun.
        exclude_recenter_pairs_in_probpair: bool.
            to decide whether the pairs in the problem pairs including recentered 
            points need to be excluded.
            
        Return
        ------
            None.
        '''
        # A_villcode,B_villcode,AB_travel_time,BA_travel_time,name_A,name_B,walking_time
        rerun_pairs = pd.read_csv(
            problem_pairs_fpath, 
            dtype={'A_villcode': 'str', 'B_villcode': 'str'}
        )

        if not os.path.exists(recentered_fpath):  # generates the file
            self.get_recentered_list(
                problem_vills_fpath=problem_vills_fpath,
                out_fpath=recentered_fpath
            )

        recenter_list = pd.read_csv(recentered_fpath, dtype='str')
        recenter_list = recenter_list['VILLCODE'].tolist()

        # 1. filter the pairs including recentered villages out
        have_recenter = (rerun_pairs['A_villcode'].isin(recenter_list)) |\
                        (rerun_pairs['B_villcode'].isin(recenter_list))
        if exclude_recenter_pairs_in_probpair:
            rerun_pairs = rerun_pairs[~have_recenter]

        # was thinking only rerun the pairs with both back and forth are empty
        both_nan = (rerun_pairs['AB_travel_time'].isna()) &\
                   (rerun_pairs['BA_travel_time'].isna())

        code_cols = ['A_villcode', 'B_villcode']
        rerun_pairs = rerun_pairs[code_cols]

        if include_recenter_pairs_in_rerun:
            # 2. pair the recentered villages to all others
            complete_vills = self.__calib['VILLCODE'].astype(str)
            non_re_vills = complete_vills[~complete_vills.isin(recenter_list)]

            # 2-1. Get the full matrix of re x non-re pairs
            re_non_pairs = [
                [f"[{x}, {y}]" for y in non_re_vills]
                for x in recenter_list
            ]
            # turn into 2darray for matrix operation
            re_non_pairs = np.array(re_non_pairs)  # 57 x 1190

            rnp_df = pd.DataFrame(
                data=list(map(literal_eval, re_non_pairs.flatten())),
                columns=code_cols
            )

            # 2-2. Get only the upper triangle of re x re pairs
            re_re_pairs = [
                [f"[{x}, {y}]" for y in recenter_list]
                for x in recenter_list
            ]
            re_re_pairs = np.array(re_re_pairs)  # 57 x 57
            # no need for flatten, the following operation gives a 1d array size = (57^2 - 57) / 2
            above_diagonal = re_re_pairs[np.triu_indices_from(
                re_re_pairs, k=1)]

            rrp_df = pd.DataFrame(
                data=list(map(literal_eval, above_diagonal)),
                columns=code_cols
            )

            # 2-3. Concat with the rerun pairs
            rerun_pairs = pd.concat(
                [rerun_pairs, rnp_df, rrp_df]).reset_index(drop=True)

        rerun_pairs['A_villcode'] = rerun_pairs['A_villcode'].astype(int)
        rerun_pairs['B_villcode'] = rerun_pairs['B_villcode'].astype(int)

        # 3. Get the lon lat
        for x in ['A', 'B']:
            tmp = self.__centroids.rename(columns={
                orig_name: f'{x}_' + orig_name.lower()
                for orig_name in self.__centroids.columns
            })
            rerun_pairs = rerun_pairs.merge(
                tmp, on=f'{x}_villcode', how='left')

        # A_villcode,B_villcode,A_lon,A_lat,B_lon,B_lat
        rerun_pairs.to_csv(out_fpath, index=False)

        return


class Helper_travel_cost():
    def __init__(self, destination: str = "./JJinTP_data_TW/Routing/") -> None:
        self.path = destination

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

        # Filter the villages (bot origin and destination)
        keep_villcodes = calib_info['VILLCODE']
        folded_pair = folded_pair[folded_pair[a_col].isin(keep_villcodes)]
        folded_pair = folded_pair[folded_pair[b_col].isin(keep_villcodes)]

        # public need special treatment
        if mode == 'public':
            # if both AB and BA time are empty, set them to 3000,
            # if one of them have value, use that value.
            folded_pair[ab_time_col] = folded_pair[ab_time_col].fillna(
                folded_pair[ba_time_col])
            folded_pair[ba_time_col] = folded_pair[ba_time_col].fillna(
                folded_pair[ab_time_col])

            fake_time = 3000
            folded_pair[ab_time_col] = folded_pair[ab_time_col].fillna(
                fake_time)
            folded_pair[ba_time_col] = folded_pair[ba_time_col].fillna(
                fake_time)

            print(folded_pair[[a_col, b_col, ab_time_col, ba_time_col]])

            return

        # ===================================
        #  Start merging village to township
        # ===================================
        # NOTE: since computing Township level using the original table would
        #       be easier, we don't need matrix now
        # 1. get the weights for each villcode by population
        #    VILLCODE: cccttttttvvv (city, township, village)
        #    weight for origin: population; weight for destination: employment
        calib_info['township'] = calib_info['VILLCODE'].astype("str").str[:-3]
        calib_info['weight_A'] = calib_info['population'] / \
            calib_info.groupby('township')['population'].transform('sum')
        calib_info['weight_B'] = calib_info['employment'] / \
            calib_info.groupby('township')['employment'].transform('sum')
        weights = calib_info[['VILLCODE', 'weight_A', 'weight_B']]

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
            weights[['VILLCODE', 'weight_A']],
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
                weights[['VILLCODE', 'weight_B']],
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

        print(town_travel_times)

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

        # Drop if code ends with 00, meaning the city. eg. 100  @@新北市(臺北縣)
        town_code = town_code[~town_code['code'].str.endswith('00')]
        # keep code in range of New Taipei: 100~199 and Taipei: 2200~2299
        town_code['code'] = town_code['code'].astype(int)  # Convert to int
        cond_ntp = town_code['code'].between(100, 199)
        cond_tp = town_code['code'].between(2200, 2299)
        town_code = town_code[cond_ntp | cond_tp]

        # Base the townships needed on the Calibration data
        town_code_TP = pd.read_csv(f"{DATA_PATH}calibration_data_TP.csv")
        town_code_TP['TOWNCODE'] = town_code_TP['VILLCODE']\
            .astype("str").str[:-3]
        town_code_TP = town_code_TP[['TOWNCODE', 'TOWNNAME']]
        town_code_TP = town_code_TP.drop_duplicates(
            subset='TOWNCODE', keep='first').reset_index(drop=True)
        town_code_TP = town_code_TP.merge(
            town_code, left_on='TOWNNAME', right_on='town', how='left')

        # Rearrange columns
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
            cur_file = pd.read_csv(file_path, usecols=col_name_map.keys())

            # 1. for each file, keep only the columns above, make a year column
            #    and then stack them up
            cur_file.rename(columns=col_name_map, inplace=True)

            cur_file['year'] = y + 1911

            survey_files.append(cur_file)

        survey_df = pd.concat(survey_files, ignore_index=True)
        # here we replace empty strings with 0's, since there are no code as 0
        survey_df = survey_df.astype(str).map(lambda x: x.strip())\
            .replace('', '0').astype(int)
        # NOTE: all the data in survey_df are int64, 'code' from town_code_TP is also int64

        # check na or 0 in each column
        # check = survey_df.apply(lambda x: x.isna() | (x == 0))
        # check = check.sum()
        # print(f"NaN and 0 (empty string) check:\n{check}")

        # 2. get the "TOWNCODE" used in travel cost
        #    right join to keep only the filtered districts (since town_code_TP was filtered)
        survey_df = survey_df.merge(
            town_code_TP.rename(
                columns={"TOWNCODE": 'TOWNCODE_orig', "TOWNNAME": 'TOWNNAME_orig'}),
            left_on='Residence', right_on='code', how='right'
        )

        # Dest: need only one of work and school to be in Taipei Metropolitan
        survey_df['Dest'] = np.where(
            survey_df['Workplace'].isin(
                town_code_TP['code']), survey_df['Workplace'],
            np.where(
                survey_df['School'].isin(
                    town_code_TP['code']), survey_df['School'],
                0
            )
        )

        # Drop if there are no destination
        survey_df = survey_df[survey_df['Dest'] != 0]

        # Make sure destination fall in Taipei Metropolitan
        survey_df = survey_df.merge(
            town_code_TP.rename(
                columns={"TOWNCODE": 'TOWNCODE_dest', "TOWNNAME": 'TOWNNAME_dest'}),
            left_on='Dest', right_on='code', how='right'
        )

        # Specify the columns
        survey_df = survey_df[[
            'Age', 'Gender', 'TOWNCODE_orig', 'TOWNNAME_orig',
            'TOWNCODE_dest', 'TOWNNAME_dest', 'Daily_transport'
        ]]

        # 下面這個可以看到只有 7534 筆是跨區移動 (total應該是13280筆)
        # tmp = survey_df[survey_df['TOWNNAME_orig'] != survey_df['TOWNNAME_dest']]
        # print(tmp[['TOWNNAME_orig', 'TOWNNAME_dest']])

        # 3. Get the grouped transportation code
        survey_df['Daily_transport'] = survey_df['Daily_transport'].fillna(98)

        survey_df = survey_df.merge(
            mode_codes, left_on='Daily_transport', right_on='code', how='left'
        )

        # ====================
        #  Check distribution
        # ====================
        modes_to_check = ['mode', 'coarse_mode', 'public_private']
        for current_mode in modes_to_check:
            print(f"Current mode is: {current_mode}")
            # 3-1 overall mode distribution
            counts = survey_df[current_mode]\
                .value_counts().reset_index(name='cnt')
            counts['pct'] = (counts['cnt'] / counts['cnt'].sum()) * 100
            counts['pct'] = counts['pct'].round(1)
            print(">> Overall distribution")
            print(counts)
            print(sum(counts['cnt']), sum(counts['pct']))

            # 3-2 based on origin, get mode distribution
            # counts = survey_df.groupby('TOWNCODE_orig')[current_mode]\
            #     .value_counts().reset_index(name='cnt')
            # counts['pct'] = (counts['cnt'] / counts.groupby('TOWNCODE_orig')['cnt']
            #                 .transform('sum')) * 100
            # counts['pct'] = counts['pct'].round(1)
            # print(counts)

        # ==========================
        #  Get the travle pair data
        # ==========================
        # expected columns: 'TOWNCODE_orig', 'TOWNNAME_orig', 'TOWNCODE_dest', 'TOWNNAME_dest', 'mode', 'pct'
        #  Notice: 1. the 'mode' here should be just private and public
        #          2. if there are < 1 count of sum of counts of private and public, then drop this pair.

        # (another option) group by TOWNCODE, count the occurance of codes in Transportation_1-10
        # 但這個方法就是很難定義分母 (不可能是sample size，也許是用all non-na count)


def TDX_helper():
    h = Helper_tdx()
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
        h2 = Helper_tdx("./.vscode/")
        h2.task_generator(file_list=file_list, keys=1)
    else:
        print("Program ending...")


def travel_cost_helper():
    htc = Helper_travel_cost()
    hpt = Helper_public_travel()
    # =============================
    #  Merge Public Transport Data
    # =============================
    # TODO: is this correct way to run this function??
    # hpt.merge_public_files(source_path="output/public_things") # this generates merged_public_{time}.csv

    # ============================================================
    #  Combine the Village Level Transportation to Township Level
    # ============================================================
    # Columns: VILLCODE,COUNTYNAME,TOWNNAME,VILLNAME,area,num_est,employment,
    #     avg_wage,num_est_ls,employment_ls,total_revenue_ls,
    #     total_value_added_ls,avg_employment_ls,population,floorspace,
    #     floorspace_R,floorspace_C,avg_price_R,avg_price_C,pop_den,employ_den
    calib = pd.read_csv(f"{DATA_PATH}calibration_data_TP.csv")  # 1247
    calib = calib[['VILLCODE', 'area', 'employment',
                   'population', 'TOWNNAME', 'VILLNAME']]

    # 先用private做測試
    # htc.village_to_township(mode='public', calib_info=calib)
    # htc.village_to_township(mode='private', calib_info=calib)


def main():
    # ============
    #  Split Data
    # ============
    h = Helper_tdx()
    h.data_into_x_splits(2, "JJinTP_data_TW/public_data/", 'rerun_pairs.csv')

    # =============
    #  Survey Data
    # =============
    htc = Helper_travel_cost()
    years = list(range(98, 106))  # ROC 98 ~ 105
    # htc.process_survey(years=years)

    # =============
    #  Travel Cost
    # =============
    # travel_cost_helper()

    # =============
    #  Public Data
    # =============
    hpt = Helper_public_travel(
        calib_path=FILE_CALIB,
        centroid_path=FILE_VILL_CENTROID
    )
    # Public data examination
    # hpt._public_data_check(fpath=f"{DATA_PATH}public_data/merged_public.csv")
    # TODO: 
    # 1. can set whether to generate problem pairs and problem village list
    # 2. self define the generated file suffixes (or file name)
    # 3. the merge public function want to be able to merge given set of files. (eg. my rerun results)
    # hpt._public_data_check(fpath=f"{OUT_PATH}rerun_public_results_1.csv")
    
    # this generates rerun pairs for recentered centroids
    # hpt.get_rerun_pairs(
    #     problem_vills_fpath=f'{OUT_PATH}problem_vills.csv',
    #     problem_pairs_fpath=f'{OUT_PATH}public_problem_pairs.csv',
    #     recentered_fpath=f'{OUT_PATH}recentered_list.csv',
    #     include_recenter_pairs_in_rerun=True,
    #     exclude_recenter_pairs_in_probpair=True,
    #     out_fpath=f'{OUT_PATH}rerun_pairs.csv'
    # )

    # this generates rerun pairs for different departure time
    hpt.get_rerun_pairs(
        problem_vills_fpath=f'{OUT_PATH}problem_vills.csv',
        problem_pairs_fpath=f'{OUT_PATH}public_problem_pairs_rerun_empty.csv',
        recentered_fpath=f'{OUT_PATH}recentered_list.csv',
        include_recenter_pairs_in_rerun=False,
        exclude_recenter_pairs_in_probpair=False,
        out_fpath=f'{OUT_PATH}rerun_pairs_v2.csv'
    )


if __name__ == "__main__":
    main()
