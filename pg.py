'''
Ref: https://github.com/tdxmotc/SampleCode/blob/master/Python3/auth_TDX.py
Modified by Kai-Yuan Ke
'''

import requests
from pprint import pprint
import json
from datetime import date, timedelta
import pandas as pd

class Auth():
    def __init__(self, app_id, app_key):
        self.app_id = app_id
        self.app_key = app_key

    def get_auth_header(self):
        content_type = 'application/x-www-form-urlencoded'
        grant_type = 'client_credentials'

        return{
            'content-type' : content_type,
            'grant_type' : grant_type,
            'client_id' : self.app_id,
            'client_secret' : self.app_key
        }


class data():
    def __init__(self, app_id, app_key, auth_response):
        self.app_id = app_id
        self.app_key = app_key
        self.auth_response = auth_response

    def get_data_header(self):
        auth_JSON = json.loads(self.auth_response.text)
        access_token = auth_JSON.get('access_token')

        return{
            'authorization': 'Bearer ' + access_token,
            'Accept-Encoding': 'gzip'
        }
    

def conds_to_str(conds: dict) -> str:
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


def set_conds(coord_from: list, coord_to: list) -> dict:
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
        "origin": coord_from,
        "destination": coord_to,
        "gc": 1.0,
        "top": 1,
        "transit": [3,4,5,6,7,8,9],
        "depart": tommorrow + "T12:00:00",
        "first_mile_mode": 0,
        "first_mile_time": 10,
        "last_mile_mode": 0,
        "last_mile_time": 10
    }
    

def main():
    app_id = 'ken19505-7e99114f-1228-4355'
    app_key = '15e3a19e-bf8d-44e8-9376-011e26de5133'

    auth_url="https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"
    url = "https://tdx.transportdata.tw/api/maas/routing?"

    # 121.55598430669878,24.9788580602204
    # 121.56397728822249,24.981549180333282

    c1 = [24.9788580602204,121.55598430669878]
    c2 = [24.981549180333282,121.56397728822249]

    url += conds_to_str(set_conds(c1, c2))

    try:
        d = data(app_id, app_key, auth_response)
        data_response = requests.get(url, headers=d.get_data_header())
    except:
        a = Auth(app_id, app_key)
        auth_response = requests.post(auth_url, a.get_auth_header())
        d = data(app_id, app_key, auth_response)
        data_response = requests.get(url, headers=d.get_data_header())
    # print(auth_response)
    # pprint(auth_response.text)
    # print(data_response)
    # pprint(data_response.text)

    df = pd.DataFrame(data_response.json()['data']['routes'][0])
    breakpoint()
    # df.to_csv("test.csv", index=False)

    # print(df)

if __name__ == '__main__':
    main()
