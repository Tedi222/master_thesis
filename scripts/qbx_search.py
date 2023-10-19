from tqdm import tqdm
from dipy.segment.clustering import qbx_and_merge
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


