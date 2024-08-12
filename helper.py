import pandas as pd
from tqdm import tqdm
import numpy as np


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

def main():
    h = Helper()
    # h.get_in_pair()
    h.data_into_x_splits()
    


if __name__ == "__main__":
    main()