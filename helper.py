import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from collections import defaultdict, Counter, deque
import numpy as np
import json
import os
import re

OUT_PATH = "output/"
DATA_PATH = "JJinTP_data_TW/"
PUBLIC_PATH = DATA_PATH + 'public_data/'
PRIVATE_PATH = DATA_PATH + 'car_data/'
SURVEY_PATH = DATA_PATH + 'survey_data/'
LOG_PATH = "log/"

FILE_CALIB = "JJinTP_data_TW/calibration_data_TP.csv"
FILE_VILL_CENTROID = "JJinTP_data_TW/Routing/village_centroid_TP.csv"

class Utils():
    def __init__(self):
        pass

    def to_mat(self, data: pd.DataFrame, is_half = True, is_same = False, **kwargs):
        '''
        Assuming the data contains only the upper triangle of the matrix, and
        the travel time are different for back and forth.
        Notice that the indices are always sorted.
        '''
        a_col = kwargs['A_villcode']
        b_col = kwargs['B_villcode']

        ab_time_col = kwargs['time'] if is_same else kwargs['ab_time']
        ba_time_col = kwargs['time'] if is_same else kwargs['ba_time']

        # 1. sort the data by a_col and b_col
        data = data.sort_values(by=[a_col, b_col], ascending=[True, True])

        if not is_half:
            return data.pivot_table(
                index=a_col, columns=b_col, values=ab_time_col, fill_value=0)

        ab_matrix = data.pivot_table(
            index=a_col, columns=b_col, values=ab_time_col, fill_value=0)
        ba_matrix = data.pivot_table(
            index=b_col, columns=a_col, values=ba_time_col, fill_value=0)

        all_points = sorted(set(data[a_col]).union(data[b_col]))

        ab_matrix = ab_matrix.reindex(
            index=all_points, columns=all_points, fill_value=0)
        ba_matrix = ba_matrix.reindex(
            index=all_points, columns=all_points, fill_value=0)

        return ab_matrix + ba_matrix

    def melt_mat(self, mat, index_list, is_same=False, keep_diag=False):
        '''
        If upper triangle and lower triangle have different value, two columns
        will be generated.
        '''
        cur_offset = 0 if keep_diag else 1
        rows, cols = np.triu_indices(len(index_list), k=cur_offset)

        if is_same:
            data = {
                'row_idx': [index_list[r] for r in rows],
                'col_idx': [index_list[c] for c in cols],
                'value': mat[rows, cols]
            }
        else:
            data = {
                'row_idx': [index_list[r] for r in rows],
                'col_idx': [index_list[c] for c in cols],
                'upper_triangle': mat[rows, cols],
                'lower_triangle': mat[cols, rows]
            }

        return pd.DataFrame(data)

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
    def __init__(
            self, calib_fpath='', centroid_path='',
            public_merged_fname='', walk_fpath=''
        ) -> None:
        self.__rerun_pairs = deque()  # will be used as FIFO, append and popleft

        calib = pd.read_csv(calib_fpath, dtype={'VILLCODE': 'str'})  # 1247
        calib = calib[['VILLCODE', 'employment', 'population', 'TOWNNAME', 'VILLNAME']]
        self.__calib = calib

        # VILLCODE,lon,lat
        self.__centroids = pd.read_csv(centroid_path, dtype={"VILLCODE": 'str'})

        # walking data
        walk = pd.read_csv(
            walk_fpath,
            usecols=['id_orig', 'id_dest', 'duration'],
            dtype={'id_orig': 'str', 'id_dest': 'str'}
        )

        # make sure A_villcode < B_villcode
        walk[['A_villcode', 'B_villcode']] = pd.DataFrame(
            list(map(sorted, zip(walk['id_orig'], walk['id_dest']))),
            index=walk.index
        )
        # NOTE: the stacked data does not require this step, since the villcode
        # are already in order (in the get_missing_pairs)

        self.__walk = walk[['A_villcode', 'B_villcode', 'duration']]

        self.__pfname = public_merged_fname

    def __check_centroid_changed_cnt(self):
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
            self, source_path: str, out_path: str = '',
            start_time="6pm", file_cnt=20
        ):
        '''
        This function will merge the series of files from 1 to 'file_cnt'
        with the stated 'start_time'. Save to file if the out_path is
        provided.

        Parameters
        ----------
        source_path: str.
            The folder holds all of the TDX respond to sub-files.
        out_path: str.
            The folder we save the merged file.
        start_time: str
            This is the departure time for public transportation,
            only 6pm and 10am available.
        file_cnt: int
            The number of files needed to be merged.

        Return
        ------
            None or pd.DataFrame, which is the merged data.
        '''
        out_file = os.path.join(out_path, self.__pfname)

        if os.path.exists(out_file):
            print(
                f'The file already exists in {out_path}, skipping this step...')
            return

        file_paths = [
            os.path.join(source_path, f"travel_at_{start_time}_{s}.csv")
            for s in range(1, file_cnt+1)
        ]

        # Load, concatenate, and save as a single file
        df = pd.concat([pd.read_csv(f) for f in file_paths], axis=0)

        if not out_path:
            return df
        df.to_csv(out_file, index=False)
        return

    def get_missing_pairs(self, stacked_public_path: str, walking_fpath: str):
        '''
        This function is to get the missing pairs during the get TDX process,
        which might be because of unstable connection. As a result, we use the
        walking data to recover the missing pairs.
        '''
        # Read in the data (merged public and walking data for reference)
        stacked_fpath = os.path.join(stacked_public_path, self.__pfname)
        stacked = pd.read_csv(
            stacked_fpath,
            dtype={'A_villcode': 'str', 'B_villcode': 'str'}
        )
        column_order = stacked.columns

        walk = pd.read_csv(
            walking_fpath,
            usecols=['id_orig', 'id_dest'],
            dtype={'id_orig': 'str', 'id_dest': 'str'}
        )

        # Remove duplicate pair
        # note that the situation of (A,B) = (2,3) and (3,2) should not exist
        # because of how we constructed the travel pairs.
        # should they exist, they will be treated as the same pair, thus one
        # of them will be dropped.
        # the idea of the following code is to make sure smaller id as idA
        stacked[['idA_norm', 'idB_norm']] = pd.DataFrame(
            list(map(sorted, zip(stacked['A_villcode'], stacked['B_villcode']))),
            index=stacked.index
        )
        stacked['id_norm'] = stacked['idA_norm'] + stacked['idB_norm']
        stacked = stacked.drop_duplicates(subset=['id_norm'])

        # Compare with walking to recover the missing pairs
        walk[['idA_norm', 'idB_norm']] = pd.DataFrame(
            list(map(sorted, zip(walk['id_orig'], walk['id_dest']))),
            index=walk.index
        )
        walk['id_norm'] = walk['idA_norm'] + walk['idB_norm']

        lost_keys = set(walk['id_norm']) - set(stacked['id_norm'])
        print("There are", len(lost_keys), "missing pairs...")

        # Fill the restored pairs with Na value
        walk = walk.drop(columns=['id_norm', 'id_orig', 'id_dest'])
        stacked = stacked.drop(columns=['id_norm', 'A_villcode', 'B_villcode'])

        complete_data = walk.merge(stacked, on=['idA_norm', 'idB_norm'], how='left')

        complete_data = complete_data.rename(columns={
            "idA_norm": 'A_villcode', "idB_norm": 'B_villcode'
        })

        complete_data.sort_values(
            by=['A_villcode', 'B_villcode'], ascending=[True, True],
            inplace=True
        )

        # The following code does two things:
        # 1. give the missing pairs the coordinates
        # 2. deal with the origin and destination swap after sorting location id
        #    -> because public data A->B and B->A has different travel time
        complete_data = complete_data.merge(
            self.__centroids.rename(columns={
                "VILLCODE": 'A_villcode', "lon": 'A_lon_new', "lat": 'A_lat_new'}),
            on='A_villcode', how='left'
        )
        complete_data = complete_data.merge(
            self.__centroids.rename(columns={
                "VILLCODE": 'B_villcode', "lon": 'B_lon_new', "lat": 'B_lat_new'}),
            on='B_villcode', how='left'
        )
        # if point A not the same and not empty then swap the value of 'AB_time' and 'BA_time', else skip
        cond_swap = (
            (complete_data['A_lat'].notna()) &
            (complete_data['A_lat'] != complete_data['A_lat_new']) |
            (complete_data['A_lon'] != complete_data['A_lon_new'])
        )

        ab_cols = ['AB_travel_time', 'AB_ttl_cost', 'AB_transfer_cnt', 'AB_route']
        ba_cols = ['BA_travel_time', 'BA_ttl_cost', 'BA_transfer_cnt', 'BA_route']
        complete_data.loc[cond_swap, ab_cols + ba_cols] = complete_data.loc[cond_swap, ba_cols + ab_cols].values

        # Keep the new lon and lat
        name_map = {
            'A_lon_new': 'A_lon', 'A_lat_new': 'A_lat',
            'B_lon_new': 'B_lon', 'B_lat_new': 'B_lat'
        }
        complete_data = complete_data.drop(columns=name_map.values())
        complete_data = complete_data.rename(columns=name_map)

        complete_data = complete_data[column_order]

        # Replace the original merged files
        complete_data.to_csv(stacked_fpath, index=False)
        print("Missing pairs restored.")

    def calibrate_counties_used(self, stacked_public_path: str):
        '''
        This function use the calibration data to clean the county pairs.
        We keep only the pairs that both counties are in the calib data.
        '''
        stacked_fpath = os.path.join(stacked_public_path, self.__pfname)
        stacked = pd.read_csv(
            stacked_fpath,
            dtype={'A_villcode': 'str', 'B_villcode': 'str'}
        )

        calib_couty_list = self.__calib['VILLCODE']

        stacked = stacked[stacked['A_villcode'].isin(calib_couty_list)]
        stacked = stacked[stacked['B_villcode'].isin(calib_couty_list)]

        print(stacked.shape)

        stacked.to_csv(stacked_fpath, index=False)

    def get_rerun_pairs(
            self, data_path: str, data_fname: str,
            out_path: str, target_time: str
        ):
        '''
        This function gets pairs with *ONE* of the directions missing for
        rerunning TDX.
        '''
        data = pd.read_csv(
            os.path.join(data_path, data_fname),
            dtype={'A_villcode': 'str', 'B_villcode': 'str'}
        )

        cond = (data['AB_travel_time'].isna()) | (data['BA_travel_time'].isna())

        keep_cols = ['A_villcode', 'B_villcode', 'A_lat', 'A_lon', 'B_lat', 'B_lon']
        to_rerun = data.loc[cond, keep_cols]

        out_fpath = os.path.join(out_path, f"rerun_TDX_{target_time}.csv")
        if not os.path.exists(out_fpath):
            to_rerun.to_csv(out_fpath, index=False)
        else:
            print(f"This rerun check file already exists, skip saving...")

    def fill_with_walk(
            self, data_path: str, data_fname: str,
            out_path: str, t_limit_minute=30
        ):
        '''
        This function only cares about the entries that has walking time less
        than t_limit_minute minutes.
        '''
        lim = t_limit_minute * 60

        fpath = os.path.join(data_path, data_fname)
        data = pd.read_csv(fpath, dtype={'A_villcode': 'str', 'B_villcode': 'str'})

        walk = self.__walk
        walk = walk[walk['duration'] < lim]

        data = data.merge(walk, on=['A_villcode', 'B_villcode'], how='left')

        # Care only the rows:
        # 1. with non empty 'duration' (meaning walking time < lim)
        # 2. both directions missing
        cond = (data['duration'].notna()) & (data['AB_travel_time'].isna()) & (data['BA_travel_time'].isna())
        data.loc[cond, 'AB_travel_time'] = data.loc[cond, 'AB_travel_time'].fillna(data.loc[cond, 'duration'])
        data.loc[cond, 'BA_travel_time'] = data.loc[cond, 'BA_travel_time'].fillna(data.loc[cond, 'duration'])

        data = data.drop(columns=['duration'])

        out_fpath = os.path.join(out_path, "fill_walk.csv")
        if not os.path.exists(out_fpath):
            data.to_csv(out_fpath, index=False)
        else:
            print(f"Walk data fill file already exists, skip saving...")

    def get_manual_check(self, data_path: str, data_fname: str, out_path: str):
        '''
        This function grabs the pairs with missing values in *BOTH* directions.
        The ones with only one direction missing would be filled with the
        travel time of the opposite direction.
        '''
        data = pd.read_csv(
            os.path.join(data_path, data_fname),
            dtype={'A_villcode': 'str', 'B_villcode': 'str'}
        )

        cond = (data['AB_travel_time'].isna()) & (data['BA_travel_time'].isna())

        to_man_check = data.loc[cond]

        fpath = os.path.join(out_path, "fill_manual_check.csv")
        if not os.path.exists(fpath):
            to_man_check.to_csv(fpath, index=False)
        else:
            print(f"Manual check file already exists, skip saving...")

    def make_public_main(
            self, stacked_public_path: str, fill_file_path: str,
            fill_file_list, out_path: str
        ):
        '''
        This function reads in all of the fill files and merge with the main
        data. After merging, save the data to destination Main.
        Also, there might exist some of the pairs with missing travel time in
        one of the directions, fill them with the opposite direction.
        '''
        data_spec = {'A_villcode': 'str', 'B_villcode': 'str'}

        stacked_fpath = os.path.join(stacked_public_path, self.__pfname)
        stacked = pd.read_csv(stacked_fpath, dtype=data_spec)
        stacked['key'] = stacked['A_villcode'] + stacked['B_villcode']
        stacked.set_index('key', inplace=True)

        for ff in fill_file_list:
            cur_fpath = os.path.join(fill_file_path, ff)
            cur_fill = pd.read_csv(cur_fpath, dtype=data_spec)
            cur_fill['key'] = cur_fill['A_villcode'] + cur_fill['B_villcode']
            cur_fill.set_index('key', inplace=True)

            # since they share the same key and we want to replace the old data
            # with the fill files, update is the most efficient way
            stacked.update(cur_fill)

        # check if there are missing in both direction
        cond_miss_two = (stacked['AB_travel_time'].isna()) & \
                        (stacked['BA_travel_time'].isna())
        check1 = stacked[cond_miss_two]
        print("There are", check1.shape[0], "pairs with missing.")

        # fill missing with reverse direction
        stacked['AB_travel_time'] = stacked['AB_travel_time'].fillna(stacked['BA_travel_time'])
        stacked['BA_travel_time'] = stacked['BA_travel_time'].fillna(stacked['AB_travel_time'])

        stacked.to_csv(os.path.join(out_path, "public_travel_time.csv"), index=False)

    def update_with_walk(
            self, final_public_fpath: str
        ):
        '''
        This function is to update the public travel times that is longer than
        the walking time.
        '''
        public_data = pd.read_csv(
            final_public_fpath,
            dtype={'A_villcode': 'str', 'B_villcode': 'str'}
        )

        public_data = public_data.merge(
            self.__walk, on=['A_villcode', 'B_villcode'], how='left'
        )

        # if both directions are longer than walking time, then update
        cond = (public_data['AB_travel_time'] > public_data['duration']) &\
               (public_data['BA_travel_time'] > public_data['duration'])

        public_data.loc[cond, 'AB_travel_time'] = public_data.loc[cond, 'duration']
        public_data.loc[cond, 'BA_travel_time'] = public_data.loc[cond, 'duration']

        public_data = public_data.drop(columns=['duration'])

        public_data.to_csv(final_public_fpath, index=False)

# ==============================================================
    def _public_data_check(self, fpath, is_both_empty=False):
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
        # TODO: split the check problem and generate problem centroid into two
        #       functions
        calib_info = self.__calib

        # since both set and list are not hashable, turn the sorted list into string
        data = pd.read_csv(fpath, dtype={'A_villcode': 'str', 'B_villcode': 'str'})
        data = data[['A_villcode', 'B_villcode', 'AB_travel_time', 'BA_travel_time']]
        data['point_set'] = [",".join(map(str, sorted(pair))) for pair in zip(data['A_villcode'], data['B_villcode'])]

        walk_data = pd.read_csv(f"{PUBLIC_PATH}travel_walking.csv")
        walk_data['point_set'] = [",".join(map(str, sorted(pair))) for pair in zip(
            walk_data['id_orig'], walk_data['id_dest'])]
        walk_data = walk_data[['point_set', 'duration']]

        calib_info['name_ch'] = [t + v for t,
                                 v in zip(calib_info['TOWNNAME'], calib_info['VILLNAME'])]
        calib_info = calib_info[['VILLCODE', 'name_ch']]

        # print(data.head(5))

        # 1. Preserve the valid VILLCODE (could be ignored when merging neighborhood to counties, there's a step doing this)
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
        if is_both_empty:
            check = data[(data['AB_travel_time'].isna()) &
                        (data['BA_travel_time'].isna())]
        else:
            check = data[(data['AB_travel_time'].isna()) |
                        (data['BA_travel_time'].isna())]
        check = check[[
            'A_villcode', 'B_villcode', 'AB_travel_time', 'BA_travel_time',
            'name_A', 'name_B', 'duration'
        ]].rename(columns={'duration': 'walking_time'})

        # print(check)
        matches = re.findall(r'/([^/.]*)\.', fpath)
        fname = matches[-1]
        check.to_csv(PUBLIC_PATH+f"problem_{fname}.csv", index=False)

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

    def get_rerun_pairs_old(
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

        # 'A_villcode','B_villcode','A_lat','A_lon','B_lat','B_lon'
        rerun_pairs.to_csv(out_fpath, index=False)

        return

    def read_public_file(self, merged_fpath: str):
        '''
        This function will store the merged public file as the attribute
        for later pair update operations.
        '''
        cols_to_keep = ['A_villcode', 'B_villcode', 'AB_travel_time', 'BA_travel_time']

        if isinstance(merged_fpath, list):
            to_cat = []
            for fpath in merged_fpath:
                cur = pd.read_csv(fpath, dtype={"A_villcode": 'str', "B_villcode": 'str'})
                cur = cur[cols_to_keep]
                to_cat.append(cur)

            main_file = pd.concat(to_cat)
        else:
            main_file = pd.read_csv(merged_fpath, dtype={"A_villcode": 'str', "B_villcode": 'str'})
            main_file = main_file[cols_to_keep]

        main_file['key'] = main_file['A_villcode'] + main_file['B_villcode']
        main_file = main_file.drop_duplicates(subset='key')
        main_file = main_file.set_index('key')

        while self.__rerun_pairs:
            rerun = self.__rerun_pairs.popleft()
            # replace data in main_file (regardless the a, b villcode order)
            key1_file = rerun.set_index('key1')
            key1_file = key1_file[cols_to_keep]
            key2_file = rerun.set_index('key2')
            key2_file = key2_file[cols_to_keep]

            main_file.update(key1_file)
            main_file.update(key2_file)

        # if one of the travel time is empty, fill in with the other
        main_file['AB_travel_time'] = main_file['AB_travel_time'].fillna(main_file['BA_travel_time'])
        main_file['BA_travel_time'] = main_file['BA_travel_time'].fillna(main_file['AB_travel_time'])

        self.merged_file = main_file.reset_index(drop=True)

        print("Merged file created.")

    def add_rerun_pairs(self, rerun_fpath):
        '''
        Currently, we have three versions of rerun pairs:
        v1 is the version for recentered and empty results from the original
           file, used 12pm as departure time.
           (the recentered villages were selected based on their frequency)
        v2 is the empty results from v1, used 11:30 am as departure time.
        v3 is the empty results from v2. Notice that these pairs could not get
           any valid result from TDX, so the data here are from manually fetched
           from google maps.
        v4 => there might be v4, since I'm thinking reviewing all of the centroids
              as I notice some are still wierd.
        '''
        cur_rerun = pd.read_csv(rerun_fpath, dtype={"A_villcode": 'str', "B_villcode": 'str'})
        cur_rerun['key1'] = cur_rerun['A_villcode'] + cur_rerun['B_villcode']
        cur_rerun['key2'] = cur_rerun['B_villcode'] + cur_rerun['A_villcode']

        # 1. remove duplicate (just to double check)
        cur_rerun = cur_rerun.drop_duplicates(subset='key1')

        self.__rerun_pairs.append(cur_rerun)

        # get file name
        matches = re.findall(r'/([^/.]*)\.', rerun_fpath)

        print(f"{matches[-1]} rerun file added.")

    def merge_walking(self, fpath, out_fpath):
        walk_data = pd.read_csv(fpath, dtype={"id_orig": 'str', "id_dest": 'str'})
        walk_data = walk_data[['id_orig', 'id_dest', 'duration']]

        all_vills = sorted(set(walk_data['id_orig']).union(walk_data['id_dest']))

        # 1. make the walking and public data into a matrix with index sorted.
        util = Utils()

        walk_args = {
            'A_villcode': 'id_orig',
            'B_villcode': 'id_dest',
            'time': 'duration'
        }
        walk_mat = util.to_mat(walk_data, is_half=True, is_same=True, **walk_args)
        walk_mat = walk_mat.to_numpy()

        public_args = {
            'A_villcode': 'A_villcode',
            'B_villcode': 'B_villcode',
            'ab_time': 'AB_travel_time',
            'ba_time': 'BA_travel_time'
        }
        public_mat = util.to_mat(self.merged_file, is_half=True, is_same=False, **public_args)
        public_mat = public_mat.to_numpy()

        # 2. get the mask where public is 0.
        mask_close_vills = (public_mat == 0)
        # print(self.merged_file.isna().sum().sum())  # 3822
        # print(sum(sum(mask_close_vills)))  # 5127 (diff = 1305 = 0's on diagonal)

        # check if these pairs have walking time less than 30 min
        # masked_values = walk_mat[mask_close_vills]
        # print(np.sort(masked_values[masked_values > 1800])[::-1])  # TODO:about 170 entries

        # 3. replace public if walk time is faster.
        mask_walk_faster = public_mat > walk_mat

        mask_use_walk = mask_close_vills | mask_walk_faster

        public_mat[mask_use_walk] = walk_mat[mask_use_walk]

        # 4. turn matrix back to pairwise dataframe.
        # NOTE: the order of A_villcode and B_villcode might not be the same as original
        public_df = util.melt_mat(mat=public_mat, index_list=all_vills)
        public_df = public_df.rename(columns={
            "row_idx": 'A_villcode',
            "col_idx": 'B_villcode',
            "upper_triangle": 'AB_travel_time',
            "lower_triangle": 'BA_travel_time'
        })

        public_df.to_csv(out_fpath, index=False)

    def save_public(self, fpath):
        self.merged_file.to_csv(fpath, index=False)


class Helper_travel_cost():
    def __init__(self, calib_path='') -> None:
        calib = pd.read_csv(calib_path, dtype={'VILLCODE': 'str'})  # 1247
        calib = calib[['VILLCODE', 'employment','population', 'TOWNNAME', 'VILLNAME']]
        self.__calib = calib

    def __flawed_neighborhood_to_county(self, mode: str, calib_info: pd.DataFrame, to_file: bool = True):
        '''
        This function is to merge the village-wise travel times to county-wise
        public columns: A_villcode,B_villcode,A_lat,A_lon,B_lat,B_lon,AB_travel_time,AB_ttl_cost,AB_transfer_cnt,AB_route,BA_travel_time,BA_ttl_cost,BA_transfer_cnt,BA_route
        private columns: id_orig,lon_orig,lat_orig,id_dest,lon_dest,lat_dest,query_status,timeofday,dayofweek,date,weight_name,distance,duration

        Parameters
        ----------
        mode: str.
            only supports 'private' and 'public'

        Return
        ------
            the county-wise travel time
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

        # =========================================
        #  Start merging neighborhoods to counties
        # =========================================
        # NOTE: since computing county level using the original table would
        #       be easier, we don't need matrix now
        # 1. get the weights for each villcode by population
        calib_info['county'] = calib_info['VILLCODE'].astype("str").str[:-3]
        calib_info['weight'] = calib_info['population'] / \
            calib_info.groupby('county')['population'].transform('sum')
        weights = calib_info[['VILLCODE', 'weight']]

        # 2. A and B cols make their county code: county_A and county_B
        df['A_county'] = df[a_col].astype("str").str[:-3]
        df['B_county'] = df[b_col].astype("str").str[:-3]

        start_pt = {
            'A': {'dest': 'B', 'vill_a': a_col, 'vill_b': b_col, 'time': ab_time_col},
            'B': {'dest': 'A', 'vill_a': b_col, 'vill_b': a_col, 'time': ba_time_col}
        }

        county_travel_dfs = []

        # following steps will use A to B as example
        for start, info in start_pt.items():
            # 3. AB_time x weights mapped with A
            merged_df = df.merge(
                weights.rename(columns={'weight': 'weight_A'}),
                left_on=info['vill_a'], right_on='VILLCODE', how='left'
            )
            merged_df['weighted_time_A'] = merged_df[info['time']] * \
                merged_df['weight_A']

            # 4. for each A_county (A or B doesn't matter, should have the same amount),
            #       group by B: sum the products from prev step
            #       and then times with the weights mapped with B
            #       and then group sum by county_B
            for town in set(df[f'{start}_county']):
                town_df = merged_df[merged_df[f'{start}_county'] == town]
                # 4-1
                by_villb_df = town_df.groupby(
                    [f"{info['dest']}_county", info['vill_b']], as_index=False
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
                    f"{info['dest']}_county", as_index=False
                )['weighted_time_B'].sum()

                # Force all the dest column to be B_county (even if it was actually A)
                # since the layout of this travel data would not be the same as in village
                # this will be just A, B, time for all pairs (was A, B, AB_time, BA_time)
                result.columns = ['B_county', 'time']
                result['A_county'] = town
                # rearrange column order
                result = result[['A_county', 'B_county', 'time']]

                county_travel_dfs.append(result)

        town_travel_times = pd.concat(
            county_travel_dfs, ignore_index=True)  # 1178

        # Check for duplicate rows based on 'col1' and 'col2'
        duplicates = town_travel_times[town_travel_times.duplicated(
            subset=['A_county', 'B_county'], keep=False)]

        # print(duplicates)

        return
        if to_file:
            df.to_csv()
            return
        return df

    def neighborhood_to_county(self, mode: str, in_path: str, out_path: str):
        '''
        This function is to merge the neighborhood-wise travel times to
        county-wise. The travel time is weighted by population for departure
        and employment for destination.

        Parameters
        ----------
        mode: str.
            only supports 'private' and 'public'
        out_fpath: str.
            the file path to save the output files.

        Return
        ------
            the county-wise travel time
        '''
        calib_info = self.__calib

        mode = mode.lower()

        folded_pair = pd.read_csv(
            os.path.join(in_path, f"{mode}_travel_time.csv"),
            dtype={'A_villcode': 'str', 'B_villcode': 'str'}
        )

        a_col = 'A_villcode'
        b_col = 'B_villcode'
        ab_time_col = 'AB_travel_time'
        ba_time_col = 'BA_travel_time'

        # Filter the neighborhoods (both origin and destination)
        keep_villcodes = calib_info['VILLCODE']
        folded_pair = folded_pair[folded_pair[a_col].isin(keep_villcodes)]
        folded_pair = folded_pair[folded_pair[b_col].isin(keep_villcodes)]

        # =========================================
        #  Start merging neighborhoods to counties
        # =========================================
        # 1. get the weights for each villcode by population
        #    VILLCODE: cccttttttvvv (city, county, village)
        #    weights for neighborhoods are summed to 1 for each county
        calib_info['county'] = calib_info['VILLCODE'].str[:-3]
        calib_info['weight_A'] = calib_info['population'] / \
            calib_info.groupby('county')['population'].transform('sum')
        calib_info['weight_B'] = calib_info['employment'] / \
            calib_info.groupby('county')['employment'].transform('sum')
        weights = calib_info[['VILLCODE', 'weight_A', 'weight_B']]

        # 2. Stack A->B and B->A into long data
        new_cols = ['vill_a', 'vill_b', 'time']
        AB_df = folded_pair[[a_col, b_col, ab_time_col]]
        AB_df.columns = new_cols
        BA_df = folded_pair[[b_col, a_col, ba_time_col]]
        BA_df.columns = new_cols

        all_pair = pd.concat([AB_df, BA_df], ignore_index=True)

        all_pair['town_a'] = all_pair['vill_a'].str[:-3]
        all_pair['town_b'] = all_pair['vill_b'].str[:-3]

        # each element is the travel times from single town to all other counties
        # (including itself)
        county_travel_dfs = []

        # 3. AB_time x weights mapped to A
        merged_df = all_pair.merge(
            weights[['VILLCODE', 'weight_A']],
            left_on='vill_a', right_on='VILLCODE', how='left'
        )
        merged_df['weighted_t_A'] = merged_df['time'] * merged_df['weight_A']

        # 4. each town (A or B doesn't matter, should be the same set),
        #    4-1 group by vill_B: sum the weighted time of A
        #    4-2 and then times with the weights mapped with vill_B
        #    4-3 and then group sum by town_B
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

            # Force all the dest column to be B_county (even if it was actually A)
            # since the layout of this travel data would not be the same as in village
            # this will be just A, B, time for all pairs (was A, B, AB_time, BA_time)
            result.columns = ['town_b', 'time']
            result['town_a'] = town
            # rearrange column order
            result = result[['town_a', 'town_b', 'time']]

            county_travel_dfs.append(result)

        town_travel_times = pd.concat(
            county_travel_dfs, ignore_index=True)  # 961

        # sort town_a and town_b for clear matrix demonstration
        town_travel_times = town_travel_times.sort_values(
            by=['town_a', 'town_b'], ascending=[True, True]
        )

        town_time_mat = town_travel_times.pivot_table(
            index='town_a', columns='town_b', values='time', fill_value=0)

        # Check if there are 0 or nan in the matrix (actually still dataframe)
        has_zero = (town_time_mat == 0).any().any()
        has_nan = town_time_mat.isnull().any().any()

        has_zero_or_nan = has_zero or has_nan
        # print(has_zero_or_nan)

        if out_path:
            fpath = os.path.join(out_path, f"{mode}_county_tt_mat.csv")
            town_time_mat.to_csv(fpath, index=False)
            return
        return town_time_mat

    def process_survey(
            self, years: list, survey_raw: str,
            survey_scratch: str, out_path: str
        ):
        '''
        This function will read, clean, and stack the survey data of specific
        years.
        There will be two types of file generated:
        1. commuting_flow.csv: the count of county pairs as the commuting flow
        2. transit mode count: will have public and private mode count files

        The process is as follows:
        1. since some of the columns are coded, before processing the data, we
           read the code mapping table from the official code book
           "變數名稱說明檔_98-105.csv"
           (specifically, transit mode and counties are coded)
        2. then we keep only the counties in the calibration data
        3. I manually divide the transit modes into public and private
           (based on our discussion), and the result were saved in a file
           called "transportation_code.csv"
        4. when processing survey data,
        '''
        # From codebook
        # 1. county code at A192:B585 (NTP: 1XX, TP: 22XX)
        town_code = pd.read_csv(os.path.join(survey_raw, "變數名稱說明檔_98-105.csv"))
        town_code = town_code.iloc[191:, :2]
        town_code.columns = ['code', 'town']

        # Drop if code ends with 00, meaning the city. eg. 100  @@新北市(臺北縣)
        town_code = town_code[~town_code['code'].str.endswith('00')]
        # keep code in range of New Taipei: 100~199 and Taipei: 2200~2299
        town_code['code'] = town_code['code'].astype(int)  # Convert to int
        cond_ntp = town_code['code'].between(100, 199)
        cond_tp = town_code['code'].between(2200, 2299)
        town_code = town_code[cond_ntp | cond_tp]

        # Base the counties needed on the calibration data,
        # so calib on left when merging
        town_code_TP = self.__calib
        town_code_TP['TOWNCODE'] = town_code_TP['VILLCODE'].str[:-3]
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
        # read the manual extraction and classification of the file 變數名稱說明檔_98-105.csv
        mode_codes = pd.read_csv(
            os.path.join(survey_scratch, "transportation_code.csv"),
            dtype={'code': 'int'}
        )

        # ================
        #  Start Cleaning
        # ================
        # mapping of the columns we need.
        col_name_map = {
            'v13': 'Age',
            'v12': 'Gender',  # 1: men, 2: women
            'v2_1': 'Residence',  # use the county code
            'v3_1': 'Working_or_studying',  # 1: work, 2: study, 3: both, 98: none
            'v4A': 'Workplace',  # use the county code
            'v4B': 'School',  # use the county code
            'v5_1': 'Transportation_1',  # mode_codes for all columns below
            'v5_2': 'Transportation_2',
            'v5_3': 'Transportation_3',
            'v5_4': 'Transportation_4',
            'v5_5': 'Transportation_5',
            'v5_6': 'Transportation_6',
            'v5_7': 'Transportation_7',
            'v5_8': 'Transportation_8',
            'v5_9': 'Transportation_9',
            'v5_10': 'Transportation_10',  # 1-10 is often used
            'v6': 'Daily_transport'  # this is mostly used
        }

        survey_files = []

        for y in years:
            file_path = os.path.join(survey_raw, f"{y}年民眾日常使用運具狀況調查原始資料.csv")
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
        #    right join to filter residence/start point (align with calibrated counties)
        survey_df = survey_df.merge(
            town_code_TP.rename(
                columns={"TOWNCODE": 'A_TOWNCODE', "TOWNNAME": 'A_TOWNNAME'}),
            left_on='Residence', right_on='code', how='right'
        )

        # Dest: need only one of work or school to be in calibrated counties
        survey_df['Dest'] = np.where(
            survey_df['Workplace'].isin(town_code_TP['code']),
            survey_df['Workplace'],
            np.where(
                survey_df['School'].isin(town_code_TP['code']),
                survey_df['School'],
                0
            )
        )

        # Drop if there are no destination (== 0)
        survey_df = survey_df[survey_df['Dest'] != 0]

        # Make sure destination fall in calibrated counties
        survey_df = survey_df.merge(
            town_code_TP.rename(
                columns={"TOWNCODE": 'B_TOWNCODE', "TOWNNAME": 'B_TOWNNAME'}),
            left_on='Dest', right_on='code', how='right'
        )

        # Specify the columns
        survey_df = survey_df[[
            'Age', 'Gender', 'A_TOWNCODE', 'A_TOWNNAME',
            'B_TOWNCODE', 'B_TOWNNAME', 'Daily_transport'
        ]]

        # the code below shows only 7534 records are cross-county (total record count is 13280)
        # tmp = survey_df[survey_df['A_TOWNCODE'] != survey_df['B_TOWNCODE']]
        # print(tmp[['A_TOWNNAME', 'B_TOWNNAME']])

        # 3. Get the grouped transportation code
        survey_df['Daily_transport'] = survey_df['Daily_transport'].fillna(98)

        survey_df = survey_df.merge(
            mode_codes, left_on='Daily_transport', right_on='code', how='left'
        )

        # ====================
        #  Check distribution
        # ====================
        # modes_to_check = ['mode', 'coarse_mode', 'public_private']
        # for current_mode in modes_to_check:
        #     print(f"Current mode is: {current_mode}")
        #     # 3-1 overall mode distribution
        #     counts = survey_df[current_mode]\
        #         .value_counts().reset_index(name='cnt')
        #     counts['pct'] = (counts['cnt'] / counts['cnt'].sum()) * 100
        #     counts['pct'] = counts['pct'].round(1)
        #     print(">> Overall distribution")
        #     print(counts)
        #     print(sum(counts['cnt']), sum(counts['pct']))

            # 3-2 based on origin, get mode distribution
            # counts = survey_df.groupby('TOWNCODE_orig')[current_mode]\
            #     .value_counts().reset_index(name='cnt')
            # counts['pct'] = (counts['cnt'] / counts.groupby('TOWNCODE_orig')['cnt']
            #                 .transform('sum')) * 100
            # counts['pct'] = counts['pct'].round(1)
            # print(counts)

        # =============================
        #  Get the commuting flow data
        # =============================
        # get count of each pair
        commuting_flow = survey_df.groupby(
            ['A_TOWNCODE', 'B_TOWNCODE']
        ).size().reset_index(name='count')

        # =========================
        #  Get the mode count data
        # =========================
        # NOTE: if there are < 1 count of sum of counts of private and public, then drop this pair.
        # 1. group by A-B pairs and get the count for public and private
        pair_result = survey_df.groupby(
            ['A_TOWNCODE', 'B_TOWNCODE', 'public_private']
        ).size().reset_index(name='count')

        pair_result = pair_result.pivot(
            index=['A_TOWNCODE', 'B_TOWNCODE'],
            columns='public_private',
            values='count'
        ).fillna(0).reset_index()

        pair_result.columns.name = None  # was 'public_private'

        # 2. Turn the dataframe into matrix (but there are missing pairs)
        full_code = town_code_TP['TOWNCODE'].tolist()  # calibrated counties

        complete_index = pd.MultiIndex.from_product([full_code, full_code], names=['A_TOWNCODE', 'B_TOWNCODE'])
        result_full = pair_result.set_index(['A_TOWNCODE', 'B_TOWNCODE']).reindex(complete_index, fill_value=0).reset_index()

        # Create a matrix with A_TOWNCODE as rows and B_TOWNCODE as columns for public_count
        pub_cnt_mat = result_full.pivot(index='A_TOWNCODE', columns='B_TOWNCODE', values='public').fillna(0)
        pri_cnt_mat = result_full.pivot(index='A_TOWNCODE', columns='B_TOWNCODE', values='private').fillna(0)

        # ==============
        #  Save to file
        # ==============
        commuting_flow.to_csv(os.path.join(out_path, 'commuting_flow.csv'), index=False)
        pub_cnt_mat.to_csv(os.path.join(out_path, 'public_mode_cnt.csv'), index=False)
        pri_cnt_mat.to_csv(os.path.join(out_path, 'private_mode_cnt.csv'), index=False)



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


def main():
    # Actually, the following part should be written in python notebook,
    # because in that form, we can write notes and seperately run code blocks.

    # ==============
    #  Data for TDX
    # ==============
    # h = Helper_tdx()

    # Because retrieving data from TDX routing service has daily upper limit
    # and limit on call times per second, we have to split the pairs into
    # pieces accordingly.
    # When I first do this, the limit was
    #     daily: <= 200,000 calls
    # frequency: <= 4 calls / sec.
    # h.data_into_x_splits(2, "JJinTP_data_TW/public_data/", 'rerun_pairs.csv')


    # ==========================
    #  Make village time matrix
    # ==========================
    # util = Utils()
    # public_travel_time_df = pd.read_csv(PUBLIC_PATH+"public_travel_time.csv")
    # private_travel_time_df = pd.read_csv(PRIVATE_PATH+"private_travel_time.csv")

    # var_args = {
    #     'A_villcode': 'A_villcode',
    #     'B_villcode': 'B_villcode',
    #     'ab_time': 'AB_travel_time',
    #     'ba_time': 'BA_travel_time'
    # }
    # public_mat = util.to_mat(public_travel_time_df, is_half=True, is_same=False, **var_args)

    # var_args = {
    #     'A_villcode': 'id_orig',
    #     'B_villcode': 'id_dest',
    #     'time': 'duration'
    # }
    # private_mat = util.to_mat(private_travel_time_df, is_half=True, is_same=True, **var_args)

    # public_mat.to_csv(DATA_PATH+"public_vill_travel_mat.csv", index=False)
    # private_mat.to_csv(DATA_PATH+"private_vill_travel_mat.csv", index=False)


    # =============
    #  Public Data
    # =============
    hpt = Helper_public_travel(
        calib_path=FILE_CALIB,
        centroid_path=FILE_VILL_CENTROID
    )

    # ---- Preprocess ----
    # Stack main data
    # hpt.merge_public_files(source_path="output/public_things/") # this generates merged_public_{time}.csv

    # Get the missings in the stack public
    # hpt._get_missing_pairs(
    #     stacked_public_fpath=PUBLIC_PATH+'merged_public.csv',
    #     walking_fpath=PUBLIC_PATH+'travel_walking.csv',
    #     out_fpath=PUBLIC_PATH+'public_miss_pairs.csv'
    # )


    # ---- Start Checking result from TDX ----
    # 1. Main files
    # hpt._public_data_check(fpath=f"{PUBLIC_PATH}merged_public.csv")
    # TODO:
    # * type out all the checking and rerun steps
    # 1. can set whether to generate problem pairs and problem village list
    # 2. self define the generated file suffixes (or file name)
    # 3. the merge public function want to be able to merge given set of files. (eg. my rerun results)
    # hpt._public_data_check(fpath=f"{PUBLIC_PATH}rerun_public_results_v2_1.csv")
    # hpt._public_data_check(fpath=f"{PUBLIC_PATH}rerun_public_results_v2_2.csv")

    # This generates rerun pairs for recentered centroids
    # hpt.get_rerun_pairs(
    #     problem_vills_fpath=f'{OUT_PATH}problem_vills.csv',
    #     problem_pairs_fpath=f'{OUT_PATH}public_problem_pairs.csv',
    #     recentered_fpath=f'{OUT_PATH}recentered_list.csv',
    #     include_recenter_pairs_in_rerun=True,
    #     exclude_recenter_pairs_in_probpair=True,
    #     out_fpath=f'{OUT_PATH}rerun_pairs_v1.csv'
    # )

    # for i in [1, 2]:
    #     hpt.get_rerun_pairs(
    #         problem_vills_fpath=f'{OUT_PATH}problem_vills.csv',
    #         problem_pairs_fpath=f'{PUBLIC_PATH}problem_rerun_public_results_v2_{i}.csv',
    #         recentered_fpath=f'{OUT_PATH}recentered_list.csv',
    #         include_recenter_pairs_in_rerun=False,
    #         exclude_recenter_pairs_in_probpair=False,
    #         out_fpath=f'{PUBLIC_PATH}rerun_pairs_v3_{i}.csv'
    #     )

    # need to call TDX to get the rerun_result
    # (TODO:but currently I use batch file, need to think how to include
    # in this process)

    # 2. Missing pairs (from the main stack, after checking with walking data)
    # hpt._public_data_check(
    #     fpath=f"{PUBLIC_PATH}public_miss_pairs_results_1.csv",
    #     is_both_empty=True
    # )
    # hpt._public_data_check(
    #     fpath=f"{PUBLIC_PATH}public_miss_pairs_results_2.csv",
    #     is_both_empty=True
    # )
    # since there are not many records in both files, I manually combine them into
    # 'problem_public_miss_pairs_results.csv'
    # hpt.get_rerun_pairs(
    #     problem_vills_fpath=f'{OUT_PATH}problem_vills.csv',
    #     problem_pairs_fpath=f'{PUBLIC_PATH}problem_public_miss_pairs_results.csv',
    #     recentered_fpath=f'{OUT_PATH}recentered_list.csv',
    #     # setting both to false get the coords for exactly the problem pairs passed in.
    #     # TODO: would the following bools set to different value?
    #     include_recenter_pairs_in_rerun=False,
    #     exclude_recenter_pairs_in_probpair=False,
    #     out_fpath=f'{PUBLIC_PATH}rerun_pairs_public_miss.csv'
    # )

    # Then we get results from TDX of the rerun pairs, and manually search
    # pairs with both ways empty on the Google maps. When filling the maps
    # result, if public is possible, leave the route empty, if only walking
    # is possible, state 'walk'. File generated named 'rerun_public_miss_pairs_results.csv'

    # In sum, there are three files output in this step,
    # public_miss_pairs_results_1.csv, ..._2.csv, rerun_public_miss_pairs_results.csv

    # ---- Merges files ----
    # 1. deal with the missings (first stack 1 and 2, then update with rerun)
    # hpt.add_rerun_pairs(PUBLIC_PATH+'rerun_public_miss_pairs_results.csv')
    # hpt.read_public_file(
    #     [PUBLIC_PATH+'public_miss_pairs_results_1.csv', PUBLIC_PATH+'public_miss_pairs_results_2.csv']
    # )
    # will have some missing, those are walking time shorter than 30 min.
    # hpt.save_public(PUBLIC_PATH+'public_miss_pairs_results.csv')

    # 2. get the full public file
    # hpt.add_rerun_pairs(PUBLIC_PATH+'rerun_public_results_v1_1.csv')
    # hpt.add_rerun_pairs(PUBLIC_PATH+'rerun_public_results_v1_2.csv')
    # hpt.add_rerun_pairs(PUBLIC_PATH+'rerun_public_results_v2_1.csv')
    # hpt.add_rerun_pairs(PUBLIC_PATH+'rerun_public_results_v2_2.csv')
    # hpt.add_rerun_pairs(PUBLIC_PATH+'rerun_public_results_v3.csv')

    # hpt.read_public_file(
    #     [PUBLIC_PATH+'merged_public.csv', PUBLIC_PATH+'public_miss_pairs_results.csv']
    # )
    # hpt.merge_walking(
    #     fpath=PUBLIC_PATH+'travel_walking.csv',
    #     out_fpath=PUBLIC_PATH+'public_travel_time.csv'
    # )


    # =============
    #  Travel Cost
    # =============
    htc = Helper_travel_cost(
        calib_path=FILE_CALIB
    )

    # Survey data
    years = list(range(98, 106))  # ROC 98 ~ 105
    htc.process_survey(
        years=years,
        public_pct_out_fpath=f"{DATA_PATH}public_mode_cnt.csv",
        private_pct_out_fpath=f"{DATA_PATH}private_mode_cnt.csv"
    )

    # Combine village to county
    # for m in ['public', 'private']:
    #     htc.village_to_county(mode=m, out_fpath=f"{DATA_PATH}{m}_town_travel_mat.csv")
    #     print(f"{m} done.")

if __name__ == "__main__":
    main()
