from database.operations import get_maps, get_date_range
from database.models import Map
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def maps_per_month():
	month_maps_list = []

	start_date, end_date = get_date_range()
	end_date = datetime(start_date.year + 1, 1, 1)
	first_days = []
	current_date = start_date

	while current_date <= end_date:
		first_days.append(current_date)

		# Move to the first day of the next month
		if current_date.month == 12:
			current_date = datetime(current_date.year + 1, 1, 1)
		else:
			current_date = datetime(current_date.year, current_date.month + 1, 1)
	
	date_pairs = [(first_days[i], first_days[i + 1]) for i in range(len(first_days) - 1)]

	for start, end in date_pairs:
		maps = get_maps(start, end)
		month_maps_list.append({
			"date" : start,
			"map_count" : len(maps)
		})

	return month_maps_list

def maps_per_week():
	week_maps_list = []

	start_date, end_date = get_date_range()
	end_date = datetime(start_date.year + 1, 1, 1)
	first_days = []
	current_date = start_date

	while current_date <= end_date:
		first_days.append(current_date)

		# Move to the first day of the next week, but not beyond end_date
		current_date += timedelta(days=7)
		if current_date > end_date:
			current_date = end_date
	
	date_pairs = [(first_days[i], first_days[i + 1]) for i in range(len(first_days) - 1)]

	for start, end in date_pairs:
		maps = get_maps(start, end)
		week_maps_list.append({
			"date" : start,
			"map_count" : len(maps)
		})

	return week_maps_list


def plot_maps_per_month():
	data = maps_per_month()
	months = []
	map_count = []
	for month in data:
		months.append(month["date"])
		map_count.append(month["map_count"])
	
	plt.plot(months, map_count)
	plt.xlabel('Month')
	plt.ylabel('Maps played')
	plt.title('Map-frequency')
	plt.savefig('my_plot.png')

def plot_maps_per_week():
	data = maps_per_week()
	weeks = []
	map_count = []
	for week in data:
		weeks.append(week["date"])
		map_count.append(week["map_count"])
	
	plt.plot(weeks, map_count)
	plt.xlabel('Month')
	plt.ylabel('Maps played')
	plt.title('Map-frequency')
	plt.savefig('my_plot.png')




if __name__ == "__main__":

	plot_maps_per_week()
