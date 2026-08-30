from data.dataframe import intervals
import pandas as pd


def initialize_ids(path: str) -> list:
    df = pd.read_csv(path, usecols = ['intersection_id'])
    intersection_ids = [intersection_id for intersection_id in df['intersection_id'].dropna().unique()]

    return intersection_ids

def rl_observed_data(path: str) -> pd.DataFrame:
    raw_data = []
    for intersection_id in initialize_ids(path):
        raw_data.append(intervals(path, intersection_id))

    data = pd.concat(raw_data, ignore_index=True)
    return data

def prepare_rl_data(path: str) -> pd.DataFrame:
    data = rl_observed_data(path)
    data = data.sort_values(['intersection_id', 'signal_group', 'phase_start']).reset_index(drop=True)

    return data

def timeline(path: str) -> dict:
    data = prepare_rl_data(path)
    intersection_ids = [intersection_id for intersection_id in data['intersection_id'].dropna().unique()]
    timelines = {}

    for intersection_id in intersection_ids:
        intersection_data = data[data['intersection_id'] == intersection_id]
        signal_groups = [signal_group for signal_group in intersection_data['signal_group'].dropna().unique()]

        for signal_group in signal_groups:
            signal_timeline = intersection_data[intersection_data['signal_group'] == signal_group]
            timelines[intersection_id, signal_group] = signal_timeline

    return timelines

def rl_event(path: str):
    timelines = timeline(path)
    cycles = {}
    for key, signal_timeline in timelines.items():
        timeline_cycles = []
        cycle = []
        for _, row in signal_timeline.iterrows():
            state = row['state']
            duration = row['duration_sec']
            if state == 2 and cycle:
                timeline_cycles.append(cycle)
                cycle = []
            cycle.append((state, duration))
        if cycle:
            timeline_cycles.append(cycle)
        cycles[key] = timeline_cycles 


    return cycles
