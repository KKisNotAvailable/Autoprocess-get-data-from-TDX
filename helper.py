import pandas as pd
from tqdm import tqdm
import numpy as np
import json


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
        Using matrix manipulation is way faster than I thought.
        Excluding the file ouputting time, this code finished within about 15 sec.
        '''
        centroids = pd.read_csv(self.path+infile)

        n = centroids.shape[0]

        tt = np.array(centroids)
        tt = tt.tolist()

        with tqdm(total=(n * (n-1) // 2), desc="Working on the large matrix") as pbar:
            result = [
                [str(x + y) for y in tt] # turn the list into string to do matrix things
                for x in tt
            ]

            for x in tt:
                for _ in tt:
                    pbar.update(1)
        
        result = np.array(result)

        above_diagonal = result[np.triu_indices_from(result, k=1)]    

        df = pd.DataFrame(
            data=list(map(eval, above_diagonal)), # turn the string-ed list back
            columns=['A_VCODE', 'A_lon', 'A_lat', 'B_VCODE', 'B_lon', 'B_lat']
        )

        df[['A_VCODE', 'B_VCODE']] = df[['A_VCODE', 'B_VCODE']].astype(int)
        
        df.to_csv(self.path+outfile, index=False)
        return

    def data_into_x_splits(self, infile: str = "in_pairs.csv", x: int = 3):
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

def main():
    h = Helper()
    sub_file_cnt = 40
    # h.get_in_pair() # this generates the "in_pair.csv"
    h.data_into_x_splits(x=sub_file_cnt) # this splits in_pair.csv into sub files for multi tasking.
    
    file_list = list(range(1,3))
    h2 = Helper("./.vscode/")
    h2.task_generator(file_list=file_list, keys=1)


if __name__ == "__main__":
    main()