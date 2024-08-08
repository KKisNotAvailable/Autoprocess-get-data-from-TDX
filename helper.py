import pandas as pd
from tqdm import tqdm
import numpy as np

def get_in_pair0():
    '''
    Using nested for loop is extremely slow.
    I never finished running this code cuz I'm impatient.
    But I guess it would take around 1.5 hours?
    '''
    path = "./JJinTP_data_TW/Routing/"
    filename = "village_centroid_TP.csv"
    centroids = pd.read_csv(path+filename)

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

    df.to_csv(path+"in_pairs.csv", index=False)
    return


def get_in_pair():
    '''
    Using matrix manipulation is way faster than I thought.
    Excluding the file ouputting time, this code finished within about 15 sec.
    '''
    path = "./JJinTP_data_TW/Routing/"
    filename = "village_centroid_TP.csv"
    centroids = pd.read_csv(path+filename)

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
    
    df.to_csv(path+"in_pairs.csv", index=False)
    return


def main():
    get_in_pair()


if __name__ == "__main__":
    main()