class Time:

    def __init__(self, seconds):
        self.total_seconds = seconds

    def to_hours_minutes_seconds(self) -> (int, int, int):
        m, s = divmod(self.total_seconds, 60)
        h, m = divmod(m, 60)
        return (h, m, s)

    def to_string(self) -> str:
        h, m, s = self.to_hours_minutes_seconds()
        string = ""
        if h != 0:
            string += " {}h".format(h)
        if m != 0:
            string += " {}m".format(m)
        if s != 0:
            string += " {}s".format(s)
        if not string:  # still empty
            return "0s"
        else:
            return string