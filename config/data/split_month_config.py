import pandas as pd

train_start, train_end = '2020-01-01', '2020-01-20'
val_start, val_end = '2020-01-21', '2020-01-25'
test_start, test_end = '2020-01-26', '2020-01-31'

time_delta = pd.Timedelta(hours=18, minutes=00, seconds=00)

train_start_dt = pd.to_datetime(train_start)
train_end_dt = pd.to_datetime(train_end) + time_delta

val_start_dt = pd.to_datetime(val_start)
val_end_dt = pd.to_datetime(val_end)+ time_delta

test_start_dt = pd.to_datetime(test_start)
test_end_dt = pd.to_datetime(test_end)+ time_delta