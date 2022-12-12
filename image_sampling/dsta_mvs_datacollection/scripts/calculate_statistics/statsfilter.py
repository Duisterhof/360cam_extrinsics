import argparse
import inspect
import os
import sqlite3

def min_dist_condition(min_dist):
    return f'min_dist <= {min_dist}'

def main(args):
    con = sqlite3.connect(args.stats_db)

    invalid_conditions = [
        min_dist_condition(args.min_dist)
    ]

    invalid_records = con.execute('SELECT img_id FROM depthstats WHERE ' + ' AND '.join(invalid_conditions))

    invalid_prefixes = set()

    for record in invalid_records:
        env, traj, cam, fn = record[0].split('/')
        img_idx, _ = fn.split('_')
        invalid_prefixes.add('/'.join([env, traj, cam, img_idx]))

    num_invalid = len(invalid_prefixes)
    print(f'There are {num_invalid} invalid frames.')

    if num_invalid > 0:
        prefix_conditions = [f'img_id NOT LIKE "{prefix}%"' for prefix in invalid_prefixes]
        valid_records = con.execute('SELECT img_id FROM depthstats WHERE ' + ' AND '.join(prefix_conditions))
    else:
        valid_records = con.execute('SELECT img_id FROM depthstats WHERE ' + ' AND '.join('NOT ' + c for c in invalid_conditions))

    valid = dict()
    for record in valid_records:
        env, traj, cam, fn = record[0].split('/')
        img_idx, _ = fn.split('_')
        key = env+'/'+traj
        if key not in valid:
            valid[key] = set()
        valid[key].add(img_idx)

    for prefix, valid_idx in valid.items():
        env, traj = prefix.split('/')
        filename = os.path.join(args.data_dir, env, traj, 'filtered.txt')
        with open(filename, 'w') as f:
            f.writelines('\n'.join(str(img_idx) for img_idx in sorted(valid_idx)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=inspect.cleandoc('''Filters data based on pre-calculated statistics.
    
    Currently only filtering by minimum distance is implemented.
    Outputs filtered.txt containing a list of valid indices in each trajectory folder.
    '''), formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--min_dist', type=float, default=0.0, help='Minimum distance for valid distance images.')
    parser.add_argument('stats_db', type=str, help='Image statistics database.')
    # parser.add_argument('out_file', type=str, help='Output filename.')
    parser.add_argument('data_dir', type=str, help='Dataset directory.')
    
    args = parser.parse_args()

    main(args)
    print('Done.')