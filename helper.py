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

        with tqdm(total = n * (n-1) // 2) as pbar:
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

        tt = np.array(centroids) # 2d array
        tt = tt.tolist() # list of lists

        with tqdm(total=(n * (n-1) // 2), desc="Working on the large matrix") as pbar:
            result = [
                # the '+' here is list operation
                [str(x + y) for y in tt] # turn the list into string to do matrix things
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
            data=list(map(literal_eval, above_diagonal)), # literal_eval is a safer version of eval on string
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
            print(f'The file already exists in {OUT_PATH}, skipping this step...')
            return

        file_paths = [f"{OUT_PATH}travel_at_{start_time}_{s}.csv" for s in range(1, file_cnt+1)]

        # Load, concatenate, and save as a single file
        df = pd.concat([pd.read_csv(f) for f in file_paths])
        df.to_csv(out_file, index=False)
        return

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
            df = pd.read_csv(f"{DATA_PATH}public_data/merged_public.csv")
            a_col = 'A_villcode'
            b_col = 'B_villcode'
            ab_time_col = 'AB_travel_time'
            ba_time_col = 'BA_travel_time'
        elif mode == 'private':
            df = pd.read_csv(f"{DATA_PATH}car_data/village_centroid_TP_output_20240605143517.csv")
            a_col = 'id_orig'
            b_col = 'id_dest'
            # back and forth share the same time for driving
            ab_time_col = 'duration'
            ba_time_col = 'duration'
        else:
            raise ValueError("receive only either 'public' or 'private'.")
        print(f"Currently working on {mode}!")

        # both depart and destination need to be the valid village
        keep_villcodes = calib_info['VILLCODE']
        df = df[df[a_col].isin(keep_villcodes)]
        df = df[df[b_col].isin(keep_villcodes)]

        all_points = sorted(set(df[a_col]).union(df[b_col]))

        # ---------------------------------------------
        #  Turn our travel table back into pair matrix
        # ---------------------------------------------
        def to_mat(df: pd.DataFrame):
            # the following part could be clearer if you run t_back_to_mat() in pg.py
            ab_matrix = df.pivot_table(index=a_col, columns=b_col, values=ab_time_col, fill_value=0)
            ba_matrix = df.pivot_table(index=b_col, columns=a_col, values=ba_time_col, fill_value=0)

            # If the villcode were sorted when generating the data,
            # ab_matrix should be the upper triangle and ba_matrix should be the lower
            ab_matrix = ab_matrix.reindex(index=all_points, columns=all_points, fill_value=0)
            ba_matrix = ba_matrix.reindex(index=all_points, columns=all_points, fill_value=0)

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
                json.dump(non_diagonal_zeros, f, indent=4, separators=(',', ': '))
        
        # recording_pairs_no_travel_time(mat)

        # ===================================
        #  Start merging village to township
        # ===================================
        # NOTE: since I find that to compute Township level using the
        #       original table would be easier, we don't need matrix now
        # 1. get the weights for each villcode by population
        calib_info['township'] = calib_info['VILLCODE'].astype("str").str[:-3]
        calib_info['weight'] = calib_info['population'] / calib_info.groupby('township')['population'].transform('sum')
        weights = calib_info[['VILLCODE', 'weight']]

        # 2. A and B cols make their township code: township_A and township_B
        df['A_township'] = df[a_col].astype("str").str[:-3]
        df['B_township'] = df[b_col].astype("str").str[:-3]

        start_pt = {
            'A': {'dest': 'B', 'vill_a': a_col, 'vill_b': b_col, 'time': ab_time_col}, 
            'B': {'dest': 'A', 'vill_a': b_col, 'vill_b': a_col, 'time': ba_time_col}
        }

        township_travel = {}

        # following steps will use A to B as example
        for k, v in start_pt.items():
            # 3. AB_time x weights mapped with A
            merged_df = df.merge(
                weights.rename(columns={'weight': 'weight_A'}), 
                left_on=v['vill_a'], right_on='VILLCODE', how='left'
            )
            merged_df['weighted_time_A'] = merged_df[v['time']] * merged_df['weight_A']
            
            # 4. for each A_township (A or B doesn't matter, should have the same amount), 
            #       group by B: sum the products from prev step
            #       and then times with the weights mapped with B
            #       and then group sum by township_B
            for town in set(df[f'{k}_township']):
                town_df = merged_df[merged_df[f'{k}_township'] == town]
                # 4-1
                by_villb_df = town_df.groupby(v['vill_b'], as_index=False)['weighted_time_A'].sum()
                # 4-2
                by_villb_df = by_villb_df.merge(
                    weights.rename(columns={'weight': 'weight_B'}), 
                    left_on=v['vill_b'], right_on='VILLCODE', how='left'
                )
                by_villb_df['weighted_time_B'] = by_villb_df['weighted_time_A'] * by_villb_df['weight_B']
                # 4-3
                result = by_villb_df.groupby(f'{v['dest']}_township', as_index=False)['weighted_time_B'].sum()
                result = result['weighted_time_B']

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
        file_paths = [f"{DATA_PATH}survey_data/{y}年民眾日常使用運具狀況調查原始資料.csv" for y in years]
        

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
        h.data_into_x_splits(x=sub_file_cnt) # this splits in_pair.csv into sub files for multi tasking.
    
    remake_task = input("Do you wish to remake task?(Y/N)")
    if remake_task.lower() == 'y':
        start = int(input("Serial number of file to start: "))
        n = int(input("Number of files wish to process: "))
        file_list = list(range(start,start+n))
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
    calib = pd.read_csv(f"{DATA_PATH}calibration_data_TP.csv") # 1247
    calib = calib[['VILLCODE', 'area', 'employment', 'population']]
    
    # 先用private做測試
    # h.village_to_township(mode='public', calib_info=calib)
    h.village_to_township(mode='private', calib_info=calib)

    # public: A_villcode,B_villcode,A_lat,A_lon,B_lat,B_lon,AB_travel_time,AB_ttl_cost,AB_transfer_cnt,AB_route,BA_travel_time,BA_ttl_cost,BA_transfer_cnt,BA_route
    # private: id_orig,lon_orig,lat_orig,id_dest,lon_dest,lat_dest,query_status,timeofday,dayofweek,date,weight_name,distance,duration
    # calibration: VILLCODE,COUNTYNAME,TOWNNAME,VILLNAME,area,num_est,employment,avg_wage,num_est_ls,employment_ls,total_revenue_ls,total_value_added_ls,avg_employment_ls,population,floorspace,floorspace_R,floorspace_C,avg_price_R,avg_price_C,pop_den,employ_den
    
    # =============
    #  Survey Data
    # =============
    years = list(range(98, 106)) # ROC 98 ~ 105
    # h.process_survey(years=years)

def main():
    travel_cost_helper()


if __name__ == "__main__":
    main()