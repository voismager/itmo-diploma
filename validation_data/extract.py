import csv
import datetime
import re
import ciso8601

if __name__ == '__main__':
    with open("access.log", "r") as f:
        with open("data.csv", "w") as out:
            writer = csv.writer(out)

            counter = 0
            current_time = None

            for line in f:
                time_string = re.findall('\[(.*)\]', line)[0]

                day = time_string[0:2]
                month = 1
                year = 2019

                hour = time_string[12:14]
                minute = time_string[15:17]
                second = time_string[18:20]

                if int(second) >= 30:
                    second = "30"
                else:
                    second = "00"

                # 2022-05-07T16:04:35+00:00
                time_string = f"{year}-01-{day} {hour}:{minute}:{second}"

                time = ciso8601.parse_datetime(time_string)

                if current_time is None:
                    current_time = time
                    counter += 1
                else:
                    if current_time == time:
                        counter += 1
                    else:
                        writer.writerow([counter])
                        current_time = time
                        counter = 0


