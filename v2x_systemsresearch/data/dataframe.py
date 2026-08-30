import pandas as pd
import json

def initialize_data(path: str) -> pd.DataFrame: 
    df = pd.read_csv(
        path,
        usecols=[
            'tmstp_utc',
            'intersection_id',
            'intersection_name',
            'intersection_states',
        ],
        parse_dates=['tmstp_utc']
    )

    return df

def adjust_data(signal_string):
    groups = json.loads(signal_string)

    row = {}

    for group in groups:
        try: 

            signal_group = group["signalGroup"]

            event = group["state-time-speed"][0]

            if event['eventState'] == 'unavailable':
                continue

            row[f"sg{signal_group}_state"] = event["eventState"]  
            row[f"sg{signal_group}_min_end"] = event["timing"]["minEndTime"]
            row[f"sg{signal_group}_max_end"] = event["timing"]["maxEndTime"] if event['timing'].get('maxEndTime') is not None else event['timing']['minEndTime']
           
        except KeyError:
            print('missing value for event', event)

    return pd.Series(row)

def useable_df(path: str, intersection_id: int = None) -> pd.DataFrame:
    df = initialize_data(path)

    if intersection_id is not None:
        df = df[df['intersection_id'] == intersection_id].copy()

    signal_data = df['intersection_states'].apply(adjust_data)
    df = pd.concat([df.drop(columns = ['intersection_states']), signal_data], axis = 1)

    return df


def intersection_df(path: str, intersection_id: int) -> pd.DataFrame:
    signal_group_names = {
        1: 'southbound_left',
        2: 'northbound_through',
        3: 'westbound_left',
        4: 'eastbound_through',
        5: 'northbound_left',
        6: 'southbound_through',
        7: 'eastbound_left',
        8: 'westbound_through'
    }
    df = useable_df(path, intersection_id)
    intersection = (df.sort_values('tmstp_utc').copy())

    transitions = []
    for sg, movement in signal_group_names.items():
        state_col = f'sg{sg}_state'
        min_col = f"sg{sg}_min_end"
        max_col = f"sg{sg}_max_end"

        valid_df = intersection.dropna(subset=[state_col]).copy()
        changed = valid_df[valid_df[state_col] != valid_df[state_col].shift()].copy()
        changed = changed[["tmstp_utc", state_col, min_col, max_col]]
        changed["signal_group"] = sg
        changed['movement'] = movement

        changed = changed.rename(columns={
                    state_col: "state",
                    min_col: "min_end",
                    max_col: "max_end"
                })

        transitions.append(changed)

    return pd.concat(transitions).sort_values("tmstp_utc").pivot(index = 'tmstp_utc', columns = 'signal_group', values = 'state').rename(columns=lambda sg: f"sg{sg}_state").ffill().reset_index()

def intervals(path: str, intersection_id: int) -> pd.DataFrame:
    MAP = {
        'stop-And-Remain': 0,
        'permissive-clearance': 1,
        'protected-Movement-Allowed': 2
    }

    df = intersection_df(path, intersection_id)

    groups = []
    for sg in range(1, 9):
        state_col = f"sg{sg}_state"
        if state_col not in df.columns:
            continue

        duration_df = df[['tmstp_utc', state_col]].copy()
        duration_df = duration_df[duration_df[state_col].isin(MAP)]
        duration_df['changed'] = duration_df[state_col] != duration_df[state_col].shift()

        duration_df = duration_df[duration_df["changed"]].copy()
        duration_df["phase_start"] = duration_df["tmstp_utc"]
        duration_df['phase_end'] = duration_df['tmstp_utc'].shift(-1)

        duration_df['duration_sec'] = (duration_df['phase_end'] - duration_df['phase_start']).dt.total_seconds()

        duration_df['state'] = duration_df[state_col].map(MAP)
        duration_df['signal_group'] = sg
        duration_df['intersection_id'] = intersection_id

        duration_df = duration_df[
                    [
                        "intersection_id",
                        "signal_group",
                        "phase_start",
                        "phase_end",
                        "state",
                        "duration_sec",
                    ]
                ]
        groups.append(duration_df)
    intervals = pd.concat(groups, ignore_index=True)
    intervals = intervals.dropna(subset=["phase_end", "duration_sec"])

    return intervals
    


