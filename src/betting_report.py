import pandas as pd
from pathlib import Path

root_dir = Path(__file__).parent.parent.resolve()

def betting_performance_by_date():
    df = pd.read_csv(root_dir / 'data' / 'betting_perf.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    start_date = df['Date'].dt.date.min()
    end_date = df['Date'].dt.date.max()
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
    daily_totals['Cumulative'] = daily_totals['Total P/L'].cumsum()

    # Reset index to have the date as a column
    daily_totals.reset_index(inplace=True)
    daily_totals['Date'] = daily_totals['Date'].dt.date  # Convert dates back to date format

    daily_totals.to_csv('csv/betting_perf_by_date.csv')

def betting_stats():
    df = pd.read_excel(root_dir / 'data' / 'betting_perf.xlsx')
    df['W/L'] = df['W/L'].str.strip()

    underdog_stats = {'bets': 0, 'wins': 0, 'loss': 0, 'profit': 0}
    overdog_stats = {'bets': 0, 'wins': 0, 'loss': 0, 'profit': 0}

    # Split 'Odds' columns into 4 separate columns
    odds = df['Book Odds'].str.split(':', expand=True).astype(float)
    gen_odds = df['Gen. Odds'].str.split(':', expand=True).astype(float)

    # Identify underdog and overdog bets
    dfo = pd.concat([odds, gen_odds], axis=1)
    dfo.columns = ['book_odds_1', 'book_odds_2', 'gen_odds_1', 'gen_odds_2']
    underdog_mask = ((dfo['book_odds_1'] > dfo['gen_odds_1']) & (dfo['book_odds_1'] > 2)) | \
                    ((dfo['book_odds_2'] > dfo['gen_odds_2']) & (dfo['book_odds_2'] > 2))
    overdog_mask  = ((dfo['book_odds_1'] > dfo['gen_odds_1']) & (dfo['book_odds_1'] < 2)) | \
                    ((dfo['book_odds_2'] > dfo['gen_odds_2']) & (dfo['book_odds_2'] < 2))

    underdog_bets = df.loc[underdog_mask, :]
    overdog_bets = df.loc[overdog_mask, :]

    # Print stats
    underdog_stats['bets'] = len(underdog_bets)
    underdog_stats['wins'] = len(underdog_bets[underdog_bets['W/L'] == 'W'])
    underdog_stats['loss'] = len(underdog_bets[underdog_bets['W/L'] == 'L'])
    underdog_stats['profit'] = underdog_bets['Amount'].sum()
    overdog_stats['bets'] = len(overdog_bets)
    overdog_stats['wins'] = len(overdog_bets[overdog_bets['W/L'] == 'W'])
    overdog_stats['loss'] = len(overdog_bets[overdog_bets['W/L'] == 'L'])
    overdog_stats['profit'] = overdog_bets['Amount'].sum()

    print(f"Underdog Stats: {underdog_stats}")
    print(f"Overdog Stats: {overdog_stats}")

betting_stats()