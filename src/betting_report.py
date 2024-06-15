import pandas as pd
from datetime import timedelta

# Read the data into a DataFrame
df = pd.read_csv('csv/betting_perf.csv')

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Determine the first and last date
start_date = df['Date'].dt.date.min()
end_date = df['Date'].dt.date.max()

# Create a date range for all dates between start and end
date_range = pd.date_range(start_date, end_date)

# Initialize a DataFrame to store the total P/L for each day
daily_totals = pd.DataFrame(date_range, columns=['Date'])
daily_totals.set_index('Date', inplace=True)

# Group original data by date and calculate total P/L for each day
daily_pl = df.groupby(df['Date'].dt.date)['P/L'].sum()

# Reindex the group to include all days in the range, filling missing days with 0 P/L
daily_pl = daily_pl.reindex(date_range.date, fill_value=0)

# Add the total P/L to the daily_totals DataFrame
daily_totals['Total P/L'] = daily_pl.values

# Calculate the cumulative P/L
daily_totals['Cumulative'] = daily_totals['Total P/L'].cumsum()

# Reset index to have the date as a column
daily_totals.reset_index(inplace=True)
daily_totals['Date'] = daily_totals['Date'].dt.date  # Convert dates back to date format

# Output the result
daily_totals.to_csv('csv/betting_perf_by_date.csv')
