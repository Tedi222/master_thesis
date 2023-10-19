import numpy as np
from tqdm import tqdm, trange
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamlinespeed import (length, set_number_of_points)
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.tracking import metrics as tm
from dipy.segment.clustering import qbx_and_merge
import time
import logging
logger = logging.getLogger(__name__)


def check_num_of_zero_clusters(clusters):
    i = 0
    for k in clusters:
        if len(k) > 1: continue
        i += 1
    return i


def qb_thresh_search(thresholds, sft, max_iter, stepsize):
    streamlines = sft.streamlines
    zero_clusters = []
    thresholds_lst = []
    qbx_lst = []
    for i in tqdm(range(0, max_iter)):
        qbx = qbx_and_merge(streamlines, thresholds, nb_pts=256, verbose=True)
        zero_clusters.append(check_num_of_zero_clusters(qbx.clusters))
        thresholds_lst.append(thresholds)
        qbx_lst.append(qbx)

        if zero_clusters[-1] == 0 or len(qbx) <= 39:
            print(f'Stopping,\n Search conditions reached: \n'
                  f'Final number of clusters: {len(qbx)} \n'
                  f'Number of single-streamline clusters: {zero_clusters[-1]}')
            break
        elif i == max_iter-1:
            print(f'Stopping, \n'
                  f' Reached maximum number of iterations {max_iter}')

            print(f'Final number of clusters: {len(qbx)} \n'
                  f'Number of single-streamline clusters: {zero_clusters[-1]}')
            break
        else:
            thresholds = [x+stepsize for x in thresholds]

    return zero_clusters, thresholds_lst, qbx_lst


def main():
    tractogram_path = '../tck2mni_tracks.trk'

    sft = load_tractogram(tractogram_path, 'same', bbox_valid_check=True)
    thresholds = [60., 50., 40., 35.]

    zero_clusters, thresholds_lst, qbx_lst = qb_thresh_search(thresholds, sft, 1, 0)
    print(zero_clusters)
    print(thresholds_lst)
    print(qbx_lst[-1].clusters)


if __name__ == "__main__":
    main()


