from string import Template
import time
import datetime


class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    microseconds = tdelta.microseconds

    d["H"] = f"{hours:02d}"
    d["M"] = f"{minutes:02d}"
    d["S"] = f"{seconds:02d}"
    d["ms"] = f"{microseconds}"[:3]
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


class Timer(object):
    def __init__(self, description):
        self.description = description

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        self.end = time.time()
        t_delta = datetime.timedelta(seconds=self.end - self.start)
        str_delta = strfdelta(t_delta, "%H:%M:%S.%ms")
        print(f"{self.description} took: {str_delta} (H:M:S)")

