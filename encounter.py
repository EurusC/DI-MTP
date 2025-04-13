import math
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", category=Warning)


E = 0.00335
A = 6378137
bias = 10-6


def prepare(df):
    df = df[(df['Lon'] < 180) & (df['Lat'] > 0) & (df['Lon'] > 0) & (df['Lat'] < 90)]
    df = df[df['Sog'] > 5]
    df = df[df['Sog'] < 100]
    df = df[df['Cog'] < 360]
    df = df[df['Cog'] > 0]
    return df


def cal_D(lon0, lat0, lon1, lat1):
    return math.sqrt((lon1 - lon0)**2 + (lat1 - lat0)**2)


def cal_T(lon0, lat0, lon1, lat1):
    upper_res = lon1 - lon0
    under_res = lat1 - lat0
    param = 0
    if upper_res >= 0 and under_res >= 0:
        param = 0
    elif under_res < 0:
        param = 180
    elif upper_res < 0 and under_res >= 0:
        param = 360
    if under_res == 0:
        if upper_res > 0:
            return math.pi / 2
        elif upper_res < 0:
            return -math.pi / 2
        else:
            return 0
    return math.atan(upper_res / under_res) + param


def cal_r_coarse(v0, cog0, v1, cog1):
    v_x_0 = v0 * math.sin(cog0)
    v_y_0 = v0 * math.cos(cog0)
    v_x_1 = v1 * math.sin(cog1)
    v_y_1 = v1 * math.cos(cog1)
    v_x_r = v_x_1 - v_x_0
    v_y_r = v_y_1 - v_y_0
    v_r = math.sqrt(v_x_r ** 2 + v_y_r ** 2)
    param = 0
    if v_x_r >= 0 and v_y_r >= 0:
        param = 0
    elif v_y_r < 0:
        param = 180
    elif v_y_r >= 0 and v_x_r < 0:
        param = 360
    if v_y_r == 0:
        if v_x_r > 0:
            return v_r, math.pi / 2
        elif v_x_r < 0:
            return v_r, param - math.pi / 2
        else:
            return v_r, 0
    r_coarse = math.atan(v_x_r / v_y_r) + param
    return v_r, r_coarse


def cal_dcpa(lon0, lat0, v0, cog0, lon1, lat1, v1, cog1):
    D = cal_D(lon0, lat0, lon1, lat1)
    T = cal_T(lon0, lat0, lon1, lat1)
    _, r_coarse = cal_r_coarse(v0, cog0, v1, cog1)
    dcpa = D * math.sin(r_coarse - T - math.pi)
    return dcpa


def cal_tcpa(lon0, lat0, v0, cog0, lon1, lat1, v1, cog1):
    D = cal_D(lon0, lat0, lon1, lat1)
    T = cal_T(lon0, lat0, lon1, lat1)
    v_r, r_coarse = cal_r_coarse(v0, cog0, v1, cog1)
    if v_r == 0:
        tcpa = bias
    else:
        tcpa = D * math.cos(r_coarse - T - math.pi) / v_r
    return tcpa


def judge_impact(traj0, traj1, tcpa_range, dcpa_range):
    start0, end0 = traj0.index.min(), traj0.index.max()
    start1, end1 = traj1.index.min(), traj1.index.max()
    overlap = (start0 <= end1) and (end0 >= start1)
    if not overlap:
        return False
    else:
        overlap_indices = traj0.index.intersection(traj1.index)
        for index in overlap_indices:
            lon0 = traj0.at[index, 'lon']
            lat0 = traj0.at[index, 'lat']
            lon1 = traj1.at[index, 'lon']
            lat1 = traj1.at[index, 'lat']
            v0 = traj0.at[index, 'sog']
            v1 = traj1.at[index, 'sog']
            cog0 = traj0.at[index, 'cog']
            cog1 = traj1.at[index, 'cog']
            dcpa = cal_dcpa(lon0, lat0, v0, cog0, lon1, lat1, v1, cog1)
            tcpa = cal_tcpa(lon0, lat0, v0, cog0, lon1, lat1, v1, cog1)
            if max(tcpa_range) > tcpa > min(tcpa_range) and max(dcpa_range) > dcpa > min(dcpa_range):
                return True
    return False