'''
API Reference from official github: 
https://github.com/tdxmotc/SampleCode/blob/master/Python3/auth_TDX.py

Author of other parts: Kai-Yuan Ke
'''

import pandas as pd
import json
import requests
import numpy as np
from datetime import date, timedelta, datetime

class BadResponse(Exception):
    '''Bad response when retrieving'''
    def __init__(self, message) -> None:
        self.message = message
        super(BadResponse, self).__init__(message)


class TDX_retriever():
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

        self.__data = pd.DataFrame(columns=tmp_cols)

    def get_log(self):
        with open("log.txt", "w") as txt_file:
            txt_file.write(self.__log)

    def _get_auth_header(self) -> dict:
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

        return{
            'authorization': 'Bearer ' + access_token,
            'Accept-Encoding': 'gzip'
        }
    
    def _authenticate(self) -> None:
        self._auth_response = requests.post(self.auth_url, self._get_auth_header())

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

    def _get_data_response(self, coord_from: list, coord_to: list):
        self.coord_from = coord_from
        self.coord_to = coord_to

        # to make sure there's conditions available
        if not self.__conds:
            self._set_condition()

        cur_url = self.url + self._conds_to_str(self.__conds)

        if not self._auth_response:
            self._authenticate()

        return requests.get(cur_url, headers=self._get_data_header())
    
    def get_transport_result(self, coord_from: list, coord_to: list) -> list:
        '''
        Will return back and forth results from the given two coords.


        '''
        resps = [
            self._get_data_response(coord_from, coord_to),
            self._get_data_response(coord_to, coord_from)
        ]

        # this show the standard response, same as the result on its webpage.
        # print(resp1)
        # print(resp1.text)

        cur_row = [*coord_from, *coord_to]

        for resp in resps:
            # --------------
            # Error Handling
            # --------------
            if "429" in str(resp):
                raise BadResponse("API rate limit exceeded")
            
            # responses other than 429 would return a 'result' key
            # so the following is to handle bad responses other than 429.
            self.__log += (
                f"({self.__cur_i}, {self.__cur_j})" +
                str(resp) + " " +
                resp.text + "\n"
            )
            msg = (
                f"Get {str(resp)}: {resp.json()['result']}"
                f" when retrieving from (i, j)=({self.__cur_i}, {self.__cur_j})"
            )
            if "200" not in str(resp):
                msg = "Bad Response! " + msg
                raise BadResponse(msg)
            
            # try to fetch data from responses, if fail, means there's
            # no public transportation needed between the given coords.
            # i.e. distance should be short in general (as least for our
            # case within the Taipei city)
            # Therefore, return empty time and route for such pairs.
            try:
                df = pd.DataFrame(resp.json()['data']['routes'][0])
                df['transport_mode'] = [d['transport']['mode'] for d in df['sections']]
                df['mode_duration'] = [d['travelSummary']['duration'] for d in df['sections']]

                cur_row.extend([
                    df['travel_time'].iloc[0], df['transfers'].iloc[0],
                    [[m, t] for m, t in zip(df['transport_mode'], df['mode_duration'])]
                ])
            except:
                self.get_log()
                cur_row.extend([np.nan, np.nan, []])
                
        return cur_row

    def get_pairwise(self, coord_list: pd.DataFrame) -> pd.DataFrame:
        # TODO: check if there's VILLCODE in the list before
        # actually adding that into data.
        n = coord_list.shape[0]
        for i in range(n):
            A_villcode = coord_list['VILLCODE'].iloc[i]
            A_coord = [coord_list['lat'].iloc[i], coord_list['lon'].iloc[i]]
            
            self.__cur_i = i
            for j in range(i+1, n):
                B_villcode = coord_list['VILLCODE'].iloc[j]
                B_coord = [coord_list['lat'].iloc[j], coord_list['lon'].iloc[j]]
                self.__cur_j = j

                new_row = []
                if self.__add_villcode:
                    # TODO: change the list to be extended
                    new_row.extend([A_villcode, B_villcode])
                new_row.extend(self.get_transport_result(A_coord, B_coord))

                self.__data.loc[len(self.__data)] = new_row

        return self.__data


def test_single(TDX: TDX_retriever):
    c1 = [24.9788580602204,121.55598430669878]
    c2 = [24.98197093297431,121.55628472119042]

# 63000080031,121.55598430669878,24.9788580602204
# 63000080032,121.56397728822249,24.981549180333282
# 63000080041,121.55628472119042,24.98197093297431

    cols = [
        'A_lat', 'A_lon', 'B_lat', 'B_lon',
        'AB_travel_time', 'AB_transfer_cnt', 'AB_route',
        'BA_travel_time', 'BA_transfer_cnt', 'BA_route'
    ]
    df = pd.DataFrame(columns=cols)
    df.loc[len(df)] = TDX.get_transport_result(c1, c2)

    # print(df)
    df.to_csv('single_pt_demo.csv', index=False)

def test_multi(TDX: TDX_retriever):
    path = ".\\JJinTP_data_TW\\Routing\\"
    filename = "village_centroid_TP.csv"
    centroids = pd.read_csv(path+filename)

    # shorter for testing
    centroids = centroids.iloc[:2]

    time_table = {
        "10am": "T10:00:00",
        "6pm": "T18:00:00"
    }

    # 現在不同打的時間是分兩個檔案，但說不定之後老師會想要合併同個檔案看??
    for k, t in time_table.items():
        TDX._set_condition(target_time=t)
        df = TDX.get_pairwise(centroids)
        df.to_csv(f'multi_pt_demo_{k}.csv', index=False)

def main():
    print(datetime.now())
    app_id = 'ken19505-7e99114f-1228-4355'
    app_key = '15e3a19e-bf8d-44e8-9376-011e26de5133'

    TDX = TDX_retriever(app_id, app_key)

    # ===================================================
    # Get single point DEMO: using get_transport_result()
    # ===================================================
    test_single(TDX)

    # ===================================================
    # Get pairwise from CSV
    # ===================================================
    # test_multi(TDX)


if __name__ == "__main__":
    main()
    