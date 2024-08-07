from datetime import date, timedelta
import json
import threading
import time

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
    

def main():
    c1 = [24.9788580602204,121.55598430669878]
    c2 = [24.981549180333282,121.56397728822249]

    # cond = conds_to_str(set_conds(c1, c2))

    # with open('env/api_key.json') as f:
    #     keys = json.load(f)

    # print(keys)

    # print(cond)

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
    
if __name__ == '__main__':
    main()
