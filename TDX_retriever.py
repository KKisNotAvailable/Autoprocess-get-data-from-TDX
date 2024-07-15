import pandas as pd
import json
import requests


class TDX_retriever():
    def __init__(
            self, 
            app_id, app_key,
            coord_from, coord_to,
            auth_url="https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token",
            url="https://tdx.transportdata.tw/api/maas/routing?"
        ):
        self.__app_id = app_id
        self.__app_key = app_key
        self.coord_from = coord_from
        self.coord_to = coord_to
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
    
    def authenticate(self) -> None:
        self._auth_response = requests.post(self.auth_url, self._get_auth_header())

    def _set_condition(self) -> str:
        pass

    def get_data_response(self):
        self._set_condition()
        return requests.get(self.url, headers=self._get_data_header())
    
    # a = Auth(app_id, app_key)
    # auth_response = requests.post(auth_url, a.get_auth_header())
    # d = data(app_id, app_key, auth_response)
    # data_response = requests.get(url, headers=d.get_data_header())
    