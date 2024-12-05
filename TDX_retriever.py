'''
API Reference from official github: 
https://github.com/tdxmotc/SampleCode/blob/master/Python3/auth_TDX.py

Author of other parts: Kai-Yuan Ke
'''
import sys
import os
import pandas as pd
import json
import requests
import numpy as np
from tqdm import tqdm
from datetime import date, timedelta, datetime
import time

# TODO:
# 1. change the way writing to log.
# 2. when single testing, should also be able to use self.__data to store result.


class BadResponse(Exception):
    '''
    Bad response when retrieving
    200: good response but with no result. (might be distance too short)
    400: bad request.
    401: invalid auth (token expire will throw this).
    404: spec params is wrong.
    429: over limited frequency per second.

    But for 404, when I encounter that, it was possibly because of
    some limitation? but not token access tho, since one program call use the 
    same access token.
    Since I tested on some points reported to trigger 404, 
    and that point actually worked, so not sure why 404 occured.
    '''

    def __init__(self, message) -> None:
        self.message = message
        super(BadResponse, self).__init__(message)


class MockResponse:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text

    def __str__(self):
        return f"<Response [{self.status_code}]>"


class TDX_retriever():
    def __init__(
        self,
        app_id, app_key,
        log_path: str = "./log/",
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
        self.__log_path = log_path
        self.__log_file = ""

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        tmp_cols = []
        if self.__add_villcode:
            tmp_cols.extend(['A_villcode', 'B_villcode'])
        tmp_cols.extend([
            'A_lat', 'A_lon', 'B_lat', 'B_lon',
            'AB_travel_time', 'AB_ttl_cost', 'AB_transfer_cnt', 'AB_route',
            'BA_travel_time', 'BA_ttl_cost', 'BA_transfer_cnt', 'BA_route'
        ])

        self.__data = pd.DataFrame(columns=tmp_cols)

    def set_log_name(self, log_name: str):
        self.__log_file = self.__log_path + log_name
        if not os.path.exists(self.__log_file):
            with open(self.__log_file, 'w') as f:
                pass  # Just create an empty file

    def write_log(self, message):
        '''
        Currently, writes log when response not 200 occurs and end of the program
        Both would write all the no data case and all the bad responses (some coord pair might overlap)
        
        index of the coords starts with 0, 1, 2, ...
        '''
        with open(self.__log_file, 'a') as f:
            f.write(f"{message}\n")

    def __get_auth_header(self) -> dict:
        content_type = 'application/x-www-form-urlencoded'
        grant_type = 'client_credentials'

        return {
            'content-type': content_type,
            'grant_type': grant_type,
            'client_id': self.__app_id,
            'client_secret': self.__app_key
        }

    def _get_data_header(self) -> dict:
        auth_JSON = json.loads(self._auth_response.text)
        access_token = auth_JSON.get('access_token')

        return {
            'authorization': 'Bearer ' + access_token,
            'Accept-Encoding': 'gzip'
        }

    def _authenticate(self) -> None:
        self._auth_response = requests.post(
            self.auth_url, self.__get_auth_header())

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
            last_mile_mode: 0 | transportation method for the last mile, (0:走路,1:腳踏車,2:開車,3:共享單車)
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
        days_to_add = [7, 1, 2, 3, 4, 5, 6]
        weekday_diff = target_weekday - date.today().isoweekday()
        next_target = (
            date.today() + timedelta(days=days_to_add[weekday_diff])
        ).strftime("%Y-%m-%d")

        timing_mode = "depart" if is_depart else "arrival"

        self.__conds = {
            "origin": self.coord_from,
            "destination": self.coord_to,
            "gc": 1.0,
            "top": 1,
            "transit": [3, 4, 5, 6, 7, 8, 9],
            "transfer_time": [0, 60],
            timing_mode: next_target + target_time,
            "first_mile_mode": 0,
            "first_mile_time": 30,
            "last_mile_mode": 0,
            "last_mile_time": 30
        }
        return
    
    def cond_output(self):
        with open(f'output/condition.json', 'w', encoding='utf-8-sig') as fp:
            json.dump(
                self.__conds, fp,
                sort_keys=False, ensure_ascii=False,
                indent=4, separators=(',', ': ')
            )

    def _update_coords(self, coord_from: list, coord_to: list):
        self.__conds['origin'] = self.coord_from = coord_from
        self.__conds['destination'] = self.coord_to = coord_to

    def _auth_tester(self):
        self._authenticate()
        auth_JSON = json.loads(self._auth_response.text)
        access_token = auth_JSON.get('access_token')

        print(access_token)

    def _get_data_response(self, coord_from: list, coord_to: list):
        # to make sure there's conditions available
        if not self.__conds:
            self._set_condition()

        self._update_coords(coord_from, coord_to)

        cur_url = self.url + self._conds_to_str(self.__conds)

        if not self._auth_response:
            # print("new_token_get")
            self._authenticate()

        # if it stops here, could be response waited too long?
        # 或反過來說，如果等太久會卡在這步嗎
        try:
            resp = requests.get(cur_url, headers=self._get_data_header())
        except:
            # 真的會跑進來這裡
            resp = MockResponse(600, "No idea what happened...")
            # self.write_log(f"!!!({self.__cur_i}, {self.__cur_j})")
            # time.sleep(1)
            # resp = requests.get(cur_url, headers=self._get_data_header())

        return resp

    def get_transport_result(self, coord_from: list, coord_to: list, get_demo=False) -> list:
        '''
        Will return back and forth results from the given two coords.

        Parameters
        ----------
        coord_from: list.
            looks like [24, 121].
        coord_to: list.
            looks like [24, 121].

        Return
        ------
            [
                coord_A, coord_B,
                A2B_time_spent, A2B_transfer_times, [A2B_modes and mode_time_spent],
                B2A_time_spent, B2A_transfer_times, [B2A_modes and mode_time_spent]
            ]
        '''
        pairs = [
            [coord_from, coord_to],
            [coord_to, coord_from]
        ]

        cur_row = [*coord_from, *coord_to]

        def get_resp(p0):
            return self._get_data_response(p0[0], p0[1])

        for p in pairs:
            resp = get_resp(p)
            # --------------
            # Error Handling
            # --------------
            # 429 and 401 (when invalid token) would not return a 'result' key.
            # while "429" in str(resp):
            #     time.sleep(0.5)
            #     resp = get_resp(p)
            #     # raise BadResponse("API rate limit exceeded")

            tmp_log = (
                f"({self.__cur_i}, {self.__cur_j})" +
                str(resp) + " " + resp.text
            )

            # while "200" not in str(resp):
            #     # sleep for 1 sec, authenticate again, then get response again
            #     time.sleep(15)
            #     self._authenticate()  # would generate new access token
            #     resp = get_resp(p)
            #     # 有error是出現在288行，代表是要重新拿過token，但不知怎地新token是empty
            #     # 先把sleep時間拉長不知道有沒有用

            #     self.write_log(tmp_log)
            #     # raise BadResponse(tmp_log)

            if "200" not in str(resp): # ignore and make record
                self.write_log(">>"+tmp_log)
                cur_row.extend([np.nan, np.nan, np.nan, []])
                continue # skip current pair

            # ------------------------------------------
            # the following is dealing with response 200
            # ------------------------------------------
            if not resp.json()['data']['routes']: # response is 200 but empty result
                self.write_log(tmp_log)
                cur_row.extend([np.nan, np.nan, np.nan, []])
                continue # skip current pair

            if get_demo:
                with open(f'output/result_example.json', 'w', encoding='utf-8-sig') as fp:
                    json.dump(
                        resp.json(), fp,
                        sort_keys=False, ensure_ascii=False,
                        indent=4, separators=(',', ': ')
                    )

            df = pd.DataFrame(resp.json()['data']['routes'][0])
            transport_mode = [
                d['transport']['mode'] for d in df['sections']
            ]
            mode_duration = [
                d['travelSummary']['duration'] for d in df['sections']
            ]

            # if price does not exist, set it to 0
            price = df['total_price'].iloc[0] if 'total_price' in df.keys() else 0

            cur_row.extend([
                df['travel_time'].iloc[0],
                price,
                df['transfers'].iloc[0],
                [[m, t] for m, t in zip(transport_mode, mode_duration)]
            ])

        return cur_row

    def get_pairwise_unpaired(self, coord_list: pd.DataFrame) -> pd.DataFrame:
        '''
        This method gets the coords from a list of coords and pair them using loop
        to get the calculated public transport time between them.

        Parameters
        ----------
        coord_list: pd.DataFrame
            three columns in this dataframe: villcode, lon, lat. each row represents
            a village, the lon and lat is this village's centroid position.

        Return    
        ------
            a dataframe with pair routing results. length should be n * (n-1) / 2,
            where n is the length of the coord_list.
        '''
        # make sure the later code could run even without villcode provided.
        if 'VILLCODE' not in coord_list.columns:
            coord_list['VILLCODE'] = 0

        n = coord_list.shape[0]
        for i in tqdm(range(n)):
            A_villcode = coord_list['VILLCODE'].iloc[i]
            A_coord = [coord_list['lat'].iloc[i], coord_list['lon'].iloc[i]]

            self.__cur_i = i
            for j in range(i+1, n):
                B_villcode = coord_list['VILLCODE'].iloc[j]
                B_coord = [coord_list['lat'].iloc[j],
                           coord_list['lon'].iloc[j]]
                self.__cur_j = j

                new_row = []
                if self.__add_villcode:
                    new_row.extend([A_villcode, B_villcode])
                new_row.extend(self.get_transport_result(A_coord, B_coord))

                self.__data.loc[len(self.__data)] = new_row

        return self.__data

    def get_pairwise_paired(self, coord_list: pd.DataFrame) -> pd.DataFrame:
        '''
        This method gets the coords from a list of coords and pair them using loop
        to get the calculated public transport time between them.

        Parameters
        ----------
        coord_list: pd.DataFrame
            six columns in this dataframe: villcode, lon, lat of village A and B. 
            each row represents a pair of village.

        Return
        ------
            a dataframe with pair routing results. length should be the same 
            as the length of the coord_list.
        '''
        n = coord_list.shape[0]
        for i in tqdm(range(n)):
            A_villcode = coord_list['A_village'].iloc[i]
            A_coord = [coord_list['A_lat'].iloc[i],
                       coord_list['A_lon'].iloc[i]]

            self.__cur_i = i
            B_villcode = coord_list['B_village'].iloc[i]
            B_coord = [coord_list['B_lat'].iloc[i],
                       coord_list['B_lon'].iloc[i]]

            new_row = []
            if self.__add_villcode:
                new_row.extend([A_villcode, B_villcode])
            new_row.extend(self.get_transport_result(A_coord, B_coord))

            self.__data.loc[len(self.__data)] = new_row

        return self.__data


def test_single(TDX: TDX_retriever):
    c1 = [24.9788580602204, 121.55598430669878]
    c2 = [25.001153300303976, 121.5042516466285]

    c1 = [24.9788580602204, 121.55598430669878]
    c2 = [24.92583789587648, 121.34128216256848]


# 63000080031,121.55598430669878,24.9788580602204
# 63000080032,121.56397728822249,24.981549180333282
# 63000080041,121.55628472119042,24.98197093297431

# 65000030010,121.5042516466285,25.001153300303976
# 65000030007,121.49481580385896,25.000770213112368

    cols = [
        'A_lat', 'A_lon', 'B_lat', 'B_lon',
        'AB_travel_time', 'AB_ttl_cost', 'AB_transfer_cnt', 'AB_route',
        'BA_travel_time', 'BA_ttl_cost', 'BA_transfer_cnt', 'BA_route'
    ]
    df = pd.DataFrame(columns=cols)
    df.loc[len(df)] = TDX.get_transport_result(c1, c2, get_demo=True)

    # TDX.cond_output()

    # print(df)
    df.to_csv('output/single_pt_demo.csv', index=False)


def get_multi(
    TDX: TDX_retriever,
    centroids: pd.DataFrame,
    time_symb: str, time_format: str,
    batch_num: str = None,
    test_size: int = 0,
    out_path: str = "./output/"
):
    print(f"Start Processing Centroids with depart time at {time_symb}...")

    dt_dtr = datetime.now().strftime("%Y%m%d")
    TDX.set_log_name(
        log_name=f"{dt_dtr}_{time_symb}_{batch_num}.log"
    )

    # shorter for testing
    if test_size > 0:
        centroids = centroids.iloc[:test_size]

    if out_path[-1] != "/":
        out_path = out_path + "/"

    # 現在不同打的時間是分開檔案，但說不定之後老師會想要合併同個檔案看??
    TDX._set_condition(target_time=time_format)
    # get result from unpaired centroids file.
    if len(centroids.columns) == 3:
        df = TDX.get_pairwise_unpaired(centroids)
    # get result from paired centroids.
    else:
        df = TDX.get_pairwise_paired(centroids)

    out_name = f'{out_path}travel_at_{time_symb}'
    if batch_num:
        out_name += f"_{batch_num}"

    df.to_csv(out_name+'.csv', index=False)


def main():
    if len(sys.argv) <= 1:
        print("Please provide a file path to the api key info.")
        return

    api_filepath = sys.argv[1]
    if ".json" not in api_filepath:
        print("Should provide a api key file in json.")
        return

    with open(api_filepath) as f:
        ps_info = json.load(f)
    app_id = ps_info['app_id']
    app_key = ps_info['app_key']

    out_path = "./output/"
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    batch_num = sys.argv[2]

    # ===================================================
    # Get single point DEMO: using get_transport_result()
    # ===================================================
    # TDX = TDX_retriever(app_id, app_key)
    # test_single(TDX)
    # return

    # ===================================================
    # Get pairwise from CSV
    # ===================================================
    path = "./JJinTP_data_TW/Routing/"
    # filename = "village_centroid_TP.csv" # unpaired data
    filename = f"in_pairs_sub{batch_num}.csv"  # paired data
    centroids = pd.read_csv(path+filename)

    time_table = {
        "10am": "T10:00:00"
        # "6pm": "T18:00:00"
    }
    for k, t in time_table.items():
        TDX = TDX_retriever(app_id, app_key, add_villcode=True)
        # TDX._auth_tester()
        get_multi(
            TDX, centroids, k, t,
            batch_num=batch_num, test_size=0, out_path=out_path
        )


if __name__ == "__main__":
    main()
