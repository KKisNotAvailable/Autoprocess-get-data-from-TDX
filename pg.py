from datetime import date, timedelta
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


class request_test:
    def __init__(self, app_id, app_key) -> None:
        self.__app_id = app_id
        self.__app_key = app_key
        self.auth_url="https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token",
        self.url="https://tdx.transportdata.tw/api/maas/routing?"
        self.__conds = None
        self._auth_response = None
        self.coord_from = [121, 25]
        self.coord_to = [121, 25]
    
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
        # access_token = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJER2lKNFE5bFg4WldFajlNNEE2amFVNm9JOGJVQ3RYWGV6OFdZVzh3ZkhrIn0.eyJleHAiOjE3MjM3OTg0MzEsImlhdCI6MTcyMzcxMjAzMSwianRpIjoiMThlNWYzODYtNzRiNS00MjFmLWE1MjYtY2M4MGIzOTFmYTY0IiwiaXNzIjoiaHR0cHM6Ly90ZHgudHJhbnNwb3J0ZGF0YS50dy9hdXRoL3JlYWxtcy9URFhDb25uZWN0Iiwic3ViIjoiN2E5ZDY2NjctZTU1Zi00ODNiLWFmNjQtM2I5MDc5MDhkNWQ5IiwidHlwIjoiQmVhcmVyIiwiYXpwIjoia2VuMTk1MDUtN2U5OTExNGYtMTIyOC00MzU1IiwiYWNyIjoiMSIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJzdGF0aXN0aWMiLCJwcmVtaXVtIiwicGFya2luZ0ZlZSIsIm1hYXMiLCJhZHZhbmNlZCIsImdlb2luZm8iLCJ2YWxpZGF0b3IiLCJ0b3VyaXNtIiwiaGlzdG9yaWNhbCIsImJhc2ljIl19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJ1c2VyIjoiOGIzOGRiMmEifQ.hEZZAmJrHEe9tPjJath6P0uaIUFOdNVRCyvtxCRceXfZXGzIHBWxEH2HiugGv0mgbQMk-XgeFDsBNjSh6BSWjrugEj9M0O4Au7LJdZ4X0uSPkxpu8i6qnjyRwIiRQi7BMbvbMibF5-tD0QKwPKaYOmB4mhrNJ-fgPbWUlRBSkku4-6FRrvnWyIZpXr-WNVdFknJDc4kkgBF34yXfuVH1NThXqtt1JkkL2I-nfvUOdPBvzK_xkKoHc8kaTkhuvTRIQIC6rZalaNtLdjx9Ko-05UWubij7251LAfQpaXKA2Xbg1Ux-95kauWgIupo5DV2-fHXHB1npKgmlqW5eyprgYA'

        return{
            'authorization': 'Bearer ' + access_token,
            'Accept-Encoding': 'gzip'
        }
    
    def _authenticate(self) -> None:
        self._auth_response = requests.post(self.auth_url, self.__get_auth_header())

    def _set_condition(
            self,
            target_weekday: int = 3,
            target_time: str = "T10:00:00",
            is_depart: bool = True
        ) -> None:
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
    
    def _conds_to_str(self, conds: dict) -> str:
        cond_l = []
        for k, v in conds.items():
            if isinstance(v, list):
                v = "%2C".join(str(i) for i in v)
            elif isinstance(v, str):
                v = v.replace(":", "%3A")
            else:
                v = str(v)
            cond_l.append(k + "=" + v)
        return "&".join(cond_l)
    
    def _update_coords(self, coord_from: list, coord_to: list):
        self.__conds['origin'] = self.coord_from = coord_from
        self.__conds['destination'] = self.coord_to = coord_to

    def data_resp(self, coord_from: list, coord_to: list):

        if not self.__conds:
            self._set_condition()
            
        self._update_coords(coord_from, coord_to)

        cur_url = self.url + self._conds_to_str(self.__conds)

        if not self._auth_response:
            # print("new_token_get")
            self._authenticate()

        requests.get(cur_url, headers=self._get_data_header())

def main():
    c1 = [24.9788580602204,121.55598430669878]
    c2 = [24.981549180333282,121.56397728822249]

    # cond = conds_to_str(set_conds(c1, c2))

    with open('env/api_key.json') as f:
        keys = json.load(f)

    # print(keys)

    # print(cond)

    tester = request_test(keys['app_id'], keys['app_key'])
    tester._authenticate()
    # tester.data_resp(c1, c2)

    # print(tester.json())

    # -----------------
    # test multi thread
    # -----------------
    # def print_numbers():
    #     for i in range(5):
    #         time.sleep(1)
    #         print(f"Thread {threading.current_thread().name}: {i}")

    # # Create two threads
    # thread1 = threading.Thread(target=print_numbers, name="Thread 1")
    # thread2 = threading.Thread(target=print_numbers, name="Thread 2")

    # # Start the threads
    # thread1.start()
    # thread2.start()

    # # Wait for both threads to complete
    # thread1.join()
    # thread2.join()

    # print("Both threads have finished execution.")
    
if __name__ == '__main__':
    main()
