from datetime import date, timedelta
import numpy as np
import pandas as pd
import json
import threading
import time
import requests

def conds_to_str(conds: dict) -> str:
    cond_l = []

    for k, v in conds.items():
        if isinstance(v, list):
            # "," = "%2C"
            v = "%2C".join(str(i) for i in v)
        elif isinstance(v, str):
            v = v.replace(":", "%3A")
        else:
            v = str(v)

        cond_l.append(k + "=" + v)

    return "&".join(cond_l)


def set_conds(coord_from: list, coord_to: list) -> dict:
    # TODO: let other conditions setable as kwargs.
    days_to_add = [7,1,2,3,4,5,6]
    target_weekday = 3 # Wednesday
    weekday_diff = target_weekday - date.today().isoweekday()
    next_target = (date.today() + timedelta(days=days_to_add[weekday_diff])).strftime("%Y-%m-%d")
    
    print(next_target)

    return {
        "origin": coord_from,
        "destination": coord_to,
        "gc": 1.0,
        "top": 1,
        "transit": [3,4,5,6,7,8,9],
        "depart": next_target + "T12:00:00",
        "first_mile_mode": 0,
        "first_mile_time": 10,
        "last_mile_mode": 0,
        "last_mile_time": 10
    }


class request_test():
    def __init__(
            self, 
            app_id, app_key,
            add_villcode: bool = False,
            auth_url="https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token",
            url="https://tdx.transportdata.tw/api/maas/routing?"
        ):
        self.__app_id = app_id
        self.__app_key = app_key
        self.__add_villcode = add_villcode
        self.auth_url = auth_url
        self.url = url
        self._auth_response = None
        # TODO: need to change the way coords are set.
        self.coord_from = [121, 25]
        self.coord_to = [121, 25]
        self.__conds = None
        self.__cur_i = -1
        self.__cur_j = -1
        self.__log = ""

        tmp_cols = []
        if self.__add_villcode:
            tmp_cols.extend(['A_villcode', 'B_villcode'])
        tmp_cols.extend([
            'A_lat', 'A_lon', 'B_lat', 'B_lon',
            'AB_travel_time', 'AB_transfer_cnt', 'AB_route',
            'BA_travel_time', 'BA_transfer_cnt', 'BA_route'
        ])
    
    def __get_auth_header(self) -> dict:
        content_type = 'application/x-www-form-urlencoded'
        grant_type = 'client_credentials'

        return{
            'content-type' : content_type,
            'grant_type' : grant_type,
            'client_id' : self.__app_id,
            'client_secret' : self.__app_key
        }

    def _get_data_header(self) -> dict:
        auth_JSON = json.loads(self._auth_response.text)
        access_token = auth_JSON.get('access_token')
        access_token = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJER2lKNFE5bFg4WldFajlNNEE2amFVNm9JOGJVQ3RYWGV6OFdZVzh3ZkhrIn0.eyJleHAiOjE3MjM3OTg0MzEsImlhdCI6MTcyMzcxMjAzMSwianRpIjoiMThlNWYzODYtNzRiNS00MjFmLWE1MjYtY2M4MGIzOTFmYTY0IiwiaXNzIjoiaHR0cHM6Ly90ZHgudHJhbnNwb3J0ZGF0YS50dy9hdXRoL3JlYWxtcy9URFhDb25uZWN0Iiwic3ViIjoiN2E5ZDY2NjctZTU1Zi00ODNiLWFmNjQtM2I5MDc5MDhkNWQ5IiwidHlwIjoiQmVhcmVyIiwiYXpwIjoia2VuMTk1MDUtN2U5OTExNGYtMTIyOC00MzU1IiwiYWNyIjoiMSIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJzdGF0aXN0aWMiLCJwcmVtaXVtIiwicGFya2luZ0ZlZSIsIm1hYXMiLCJhZHZhbmNlZCIsImdlb2luZm8iLCJ2YWxpZGF0b3IiLCJ0b3VyaXNtIiwiaGlzdG9yaWNhbCIsImJhc2ljIl19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJ1c2VyIjoiOGIzOGRiMmEifQ.hEZZAmJrHEe9tPjJath6P0uaIUFOdNVRCyvtxCRceXfZXGzIHBWxEH2HiugGv0mgbQMk-XgeFDsBNjSh6BSWjrugEj9M0O4Au7LJdZ4X0uSPkxpu8i6qnjyRwIiRQi7BMbvbMibF5-tD0QKwPKaYOmB4mhrNJ-fgPbWUlRBSkku4-6FRrvnWyIZpXr-WNVdFknJDc4kkgBF34yXfuVH1NThXqtt1JkkL2I-nfvUOdPBvzK_xkKoHc8kaTkhuvTRIQIC6rZalaNtLdjx9Ko-05UWubij7251LAfQpaXKA2Xbg1Ux-95kauWgIupo5DV2-fHXHB1npKgmlqW5eyprgYA'


        return{
            'authorization': 'Bearer ' + access_token,
            'Accept-Encoding': 'gzip'
        }
    
    def _authenticate(self) -> None:
        self._auth_response = requests.post(self.auth_url, self.__get_auth_header())

    def _conds_to_str(self, conds: dict) -> str:
        '''
        The conditions are supposed to be either a list, a str, or a number.

        Parameters
        ----------
        conds: dict.
            A dictionary containing conditions of query.

        Return
        ------
        str
            Merged str of conditions, will be appended on the end of the request url.
        '''
        cond_l = []

        for k, v in conds.items():
            if isinstance(v, list):
                # "," = "%2C"
                v = "%2C".join(str(i) for i in v)
            elif isinstance(v, str):
                v = v.replace(":", "%3A")
            else:
                v = str(v)

            cond_l.append(k + "=" + v)

        return "&".join(cond_l)

    def _set_condition(
            self,
            target_weekday: int = 3,
            target_time: str = "T10:00:00",
            is_depart: bool = True
        ) -> None:
        '''
        Can set conditions manually or would set to default.
        1. "origin" and "destination" are not settable.
        2. weekday should be numeric, eg. Monday = 1 and Sunday = 7.
        3. target_time should follow the format, eg. 10am = "T10:00:00".
        4. is_depart default to be true, meaning the time provided is set as depart time.

        Some brief intro to all the parameters:
        Required:
            origin: [latitude,longitude] | start place
            destination: [latitude,longitude] | end place
            gc: 1.0 | preference of choice 0.0=cheapest, 1.0=fastest
            top: 1 | number of routes returned, default=5
            transit: [3,...,9] | ways of transportation (3:高鐵,4:台鐵,5:公車,6:捷運,7:輕軌,8:渡輪,9:纜車,20:航空)
        Optional:
            transfer_time: [0,60] | tranfer time tolerance between min=0 and max=60
            depart: "2024-07-15T12:00:00" | departure time, must be later than current time
            arrival: "2024-07-15T12:00:00" | arrival time, must be later than current time
            >> NOTE: 1. fill only depart or arrival.
                    2. the search would be based on given time but might be adjusted
                        earlier or later.
            first_mile_mode: 0 | transportation method for the first mile, (0:走路,1:腳踏車,2:開車,3:共享單車)
            first_mile_time: 30 | first mile time tolerance, max=60
            last_mile_mode: 0 | transportation method for the lsat mile, (0:走路,1:腳踏車,2:開車,3:共享單車)
            last_mile_time: 30 | last mile time tolerance, max=60

        Parameters
        ----------
        coord_from: list
        coord_to: list
        
        Return
        ------
        dict
            the dictionary of conditions.
        '''
        days_to_add = [7,1,2,3,4,5,6]
        weekday_diff = target_weekday - date.today().isoweekday()
        next_target = (date.today() + timedelta(days=days_to_add[weekday_diff])).strftime("%Y-%m-%d")
    
        timing_mode = "depart" if is_depart else "arrival"
        
        self.__conds = {
            "origin": self.coord_from,
            "destination": self.coord_to,
            "gc": 1.0,
            "top": 1,
            "transit": [3,4,5,6,7,8,9],
            "transfer_time": [0,60],
            timing_mode: next_target + target_time,
            "first_mile_mode": 0,
            "first_mile_time": 30,
            "last_mile_mode": 0,
            "last_mile_time": 30
        }
        return

    def _update_coords(self, coord_from: list, coord_to: list):
        self.__conds['origin'] = self.coord_from = coord_from
        self.__conds['destination'] = self.coord_to = coord_to

    def _auth_tester(self):
        self._authenticate()
        auth_JSON = json.loads(self._auth_response.text)
        access_token = auth_JSON.get('access_token')

        print(access_token)
    
    def data_resp(self, coord_from: list, coord_to: list):

        if not self.__conds:
            self._set_condition()
            
        self._update_coords(coord_from, coord_to)

        cur_url = self.url + self._conds_to_str(self.__conds)

        if not self._auth_response:
            # print("new_token_get")
            self._authenticate()

        return requests.get(cur_url, headers=self._get_data_header())


def t_request(c1, c2):
    with open('env/api_key1.json') as f:
        keys = json.load(f)

    print(keys)

    tester = request_test(keys['app_id'], keys['app_key'])
    resp = tester.data_resp(c1, c2)
    print(resp.text)
    # print(resp.json())


def t_multi_threads():
    def print_numbers():
        for i in range(5):
            time.sleep(1)
            print(f"Thread {threading.current_thread().name}: {i}")

    # Create two threads
    thread1 = threading.Thread(target=print_numbers, name="Thread 1")
    thread2 = threading.Thread(target=print_numbers, name="Thread 2")

    # Start the threads
    thread1.start()
    thread2.start()

    # Wait for both threads to complete
    thread1.join()
    thread2.join()

    print("Both threads have finished execution.")


def t_back_to_mat():
    data = {
        'A': ['Point1', 'Point1', 'Point2'],
        'B': ['Point2', 'Point3', 'Point3'],
        'AB_time': [10, 15, 12],
        'BA_time': [8, 14, 13]
    }
    df = pd.DataFrame(data)

    # Create a pivot table for AB_time from A to B
    ab_matrix = df.pivot_table(index='A', columns='B', values='AB_time', fill_value=0)
    

    # If you also want to include BA_time as a symmetric entry
    ba_matrix = df.pivot_table(index='B', columns='A', values='BA_time', fill_value=0)
    # Reindex both matrices to make sure they have the same structure
    all_points = sorted(set(df['A']).union(df['B']))
    ab_matrix = ab_matrix.reindex(index=all_points, columns=all_points, fill_value=0)
    ba_matrix = ba_matrix.reindex(index=all_points, columns=all_points, fill_value=0)

    print(ab_matrix)
    print(ba_matrix)

    # Combine matrices by taking the maximum of each element (or mean/any other rule you prefer)
    combined_matrix = ab_matrix + ba_matrix

    print(combined_matrix)

def t_mat_to_df():
    locations = ['locationA', 'locationB', 'locationC']
    matrix = np.array([
        [0, 60, 70],  # locationA -> others
        [54, 0, 80],  # locationB -> others
        [65, 75, 0]   # locationC -> others
    ])
    matrix = np.array([
        [0, 60, 70],  # locationA -> others
        [60, 0, 80],  # locationB -> others
        [70, 80, 0]   # locationC -> others
    ])

    # Create a DataFrame to store the data
    rows, cols = np.triu_indices(len(locations), k=1)

    data = {
        'depart': [locations[r] for r in rows],
        'dest': [locations[c] for c in cols],
        'forward': matrix[rows, cols],
        'backward': matrix[cols, rows]
    }

    df = pd.DataFrame(data)

    print(df)   


def main():
    c1 = [24.9788580602204,121.55598430669878]
    c2 = [24.981549180333282,121.56397728822249]

    # =====================
    #  T_set_TDX_condition
    # =====================
    # cond = conds_to_str(set_conds(c1, c2))
    # print(cond)

    # t_back_to_mat()

    t_mat_to_df()

    
if __name__ == '__main__':
    main()
