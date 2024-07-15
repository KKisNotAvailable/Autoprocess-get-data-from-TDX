import pandas as pd
import json
import requests
from datetime import date, timedelta


class TDX_retriever():
    def __init__(
            self, 
            app_id, app_key,
            auth_url="https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token",
            url="https://tdx.transportdata.tw/api/maas/routing?"
        ):
        self.__app_id = app_id
        self.__app_key = app_key
        self._auth_response = None
        self.auth_url = auth_url
        self.url = url

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

    def _set_condition(self) -> str:
        '''
        Some brief intro to all the parameters:
        Required:
            origin: [latitude,longitude] | start place
            destination: [latitude,longitude] | end place
            gc: 0.0 | preference of choice 0.0=cheapest, 1.0=fastest
            top: 5 | number of routes returned, default=5
            transit: [3,...] | ways of transportation (3:高鐵,4:台鐵,5:公車,6:捷運,7:輕軌,8:渡輪,9:纜車,20:航空)
        Optional:
            transfer_time: [15,60] | tranfer time tolerance between min=0 and max=60
            depart: "2024-07-15T12:00:00" | departure time, must be later than current time
            arrival: "2024-07-15T12:00:00" | arrival time, must be later than current time
            >> NOTE: 1. fill only depart or arrival.
                    2. the search would be based on given time but might be adjusted
                        earlier or later.
            first_mile_mode: 0 | transportation method for the first mile, (0:走路,1:腳踏車,2:開車,3:共享單車)
            first_mile_time: 10 | first mile time tolerance, max=60
            last_mile_mode: 0 | transportation method for the lsat mile, (0:走路,1:腳踏車,2:開車,3:共享單車)
            last_mile_time: 10 | last mile time tolerance, max=60

        Parameters
        ----------
        coord_from: list
        coord_to: list
        
        Return
        ------
        dict
            the dictionary of conditions.
        '''
        # TODO: let other conditions setable as kwargs.
        tommorrow = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

        return {
            "origin": self.coord_from,
            "destination": self.coord_to,
            "gc": 1.0,
            "top": 1,
            "transit": [3,4,5,6,7,8,9],
            "depart": tommorrow + "T08:00:00",
            "first_mile_mode": 0,
            "first_mile_time": 10,
            "last_mile_mode": 0,
            "last_mile_time": 10
        }

    def _get_data_response(self, coord_from: list, coord_to: list):
        self.coord_from = coord_from
        self.coord_to = coord_to

        cur_url = self.url + self._conds_to_str(self._set_condition())

        if not self._auth_response:
            self._authenticate()

        return requests.get(cur_url, headers=self._get_data_header())
    
    def get_transport_result(self, coord_from: list, coord_to: list):
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

        cur_row = [coord_from, coord_to]

        for resp in resps:
            df = pd.DataFrame(resp.json()['data']['routes'][0])
            df['transport_mode'] = [d['transport']['mode'] for d in df['sections']]
            df['mode_duration'] = [d['travelSummary']['duration'] for d in df['sections']]

            cur_row.extend([
                df['travel_time'].iloc[0], df['transfers'].iloc[0],
                [[m, t] for m, t in zip(df['transport_mode'], df['mode_duration'])]
            ])
                
        return cur_row


def main():
    app_id = 'ken19505-7e99114f-1228-4355'
    app_key = '15e3a19e-bf8d-44e8-9376-011e26de5133'
    c1 = [24.9788580602204,121.55598430669878]
    c2 = [24.981549180333282,121.56397728822249]

    cols = [
        'A_coord', 'B_coord', 
        'AB_travel_time', 'AB_transfer_cnt', 'AB_route',
        'BA_travel_time', 'BA_transfer_cnt', 'BA_route'
    ]
    df = pd.DataFrame(columns=cols)


    TDX = TDX_retriever(app_id, app_key)

    df.loc[len(df)] = TDX.get_transport_result(c1, c2)

    # print(df)


if __name__ == "__main__":
    main()
    