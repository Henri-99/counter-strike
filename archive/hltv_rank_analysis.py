from database.setup import session
from database.models import Match, Map
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import or_, and_, func

def year_by_year():
    years = ['2019','2020','2021','2022','2023', 'all']

    for year in years:
        query = session.query(Match)\
            .filter(and_(Match.team1_rank.isnot(None), Match.team2_rank.isnot(None)))\
            .filter(Match.best_of.isnot(2))\
            # .filter(Match.lan == 1 )
        if year != 'all':
            query = query.filter(and_(
                Match.datetime >= f"{year}-01-01",
                Match.datetime <= f"{year}-12-31"
            ))
        # .filter(and_(
        #     func.abs(Match.team1_rank - Match.team2_rank) >= 40,
        #     func.abs(Match.team1_rank - Match.team2_rank) <= 50
        # )) 

        result = query.all()
        print(f"{year.ljust(4)}: {len(result)} matches played")

        # Creating a DataFrame from the result
        columns = [
            "id", "datetime", "team1_id", "team2_id", "team1", "team2", "team1_rank", "team2_rank",
            "t1_score", "t2_score", "win"
        ]

        data = [
            (
                match.id,
                match.datetime,
                match.team1_id,
                match.team2_id,
                match.team1,
                match.team2,
                match.team1_rank,
                match.team2_rank,
                match.team1_score,
                match.team2_score,
                0 if match.team1_score < match.team2_score else 1
            )
            for match in result
        ]

        df = pd.DataFrame(data, columns=columns)
        df['expected'] = (df['team1_rank'] < df['team2_rank']).astype(int)
        percentage = (df['win'] == df['expected']).mean() * 100
        print(f"Percentage of win==expected: {percentage:.2f}%")

def rank_threshold():
    ranks = [5, 10, 20, 30]

    for rank in ranks:
        query = session.query(Match)\
            .filter(and_(Match.team1_rank.isnot(None), Match.team2_rank.isnot(None)))\
            .filter(Match.best_of.isnot(2))\
            .filter(and_(Match.team1_rank <= rank, Match.team2_rank <= rank))\
            .filter(Match.lan == 1 )
            # .filter(and_(
            #     func.abs(Match.team1_rank - Match.team2_rank) >= 40,
            #     func.abs(Match.team1_rank - Match.team2_rank) <= 50
            # )) 

        result = query.all()
        print(f"{rank}: {len(result)} matches", end = " ")

        # Creating a DataFrame from the result
        columns = [
            "id", "datetime", "team1_id", "team2_id", "team1", "team2", "team1_rank", "team2_rank",
            "t1_score", "t2_score", "win"
        ]

        data = [
            (
                match.id,
                match.datetime,
                match.team1_id,
                match.team2_id,
                match.team1,
                match.team2,
                match.team1_rank,
                match.team2_rank,
                match.team1_score,
                match.team2_score,
                0 if match.team1_score < match.team2_score else 1
            )
            for match in result
        ]

        df = pd.DataFrame(data, columns=columns)
        df['expected'] = (df['team1_rank'] < df['team2_rank']).astype(int)
        percentage = (df['win'] == df['expected']).mean() * 100
        print(f"Accuracy: {percentage:.2f}%")

def rank_differential():
    threshold = 20

    for diff in range(1,threshold):
        query = session.query(Match)\
            .filter(and_(Match.team1_rank.isnot(None), Match.team2_rank.isnot(None)))\
            .filter(Match.best_of.isnot(2))\
            .filter(and_(Match.team1_rank <= threshold, Match.team2_rank <= threshold))\
            .filter(func.abs(Match.team1_rank - Match.team2_rank) == diff) \
            # .filter(Match.lan == 1 )\

        result = query.all()
        print(f"{diff}: {len(result)} matches", end = " ")

        # Creating a DataFrame from the result
        columns = [
            "id", "datetime", "team1_id", "team2_id", "team1", "team2", "team1_rank", "team2_rank",
            "t1_score", "t2_score", "win"
        ]

        data = [
            (
                match.id,
                match.datetime,
                match.team1_id,
                match.team2_id,
                match.team1,
                match.team2,
                match.team1_rank,
                match.team2_rank,
                match.team1_score,
                match.team2_score,
                0 if match.team1_score < match.team2_score else 1
            )
            for match in result
        ]

        df = pd.DataFrame(data, columns=columns)
        df['expected'] = (df['team1_rank'] < df['team2_rank']).astype(int)
        percentage = (df['win'] == df['expected']).mean() * 100
        print(f"Accuracy: {percentage:.2f}%")


if __name__ == "__main__":
    # year_by_year()
    # rank_threshold()
    rank_differential()