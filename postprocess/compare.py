"""
compare - Comparisons between label sets
"""

# Standard Python library
from collections import namedtuple

# Add-on libraries
import numpy as np


# See compare_labels documentation for interpretation of fields
ComparisonType = namedtuple(
    "ComparisonType",
    ("detections_iou", "ground_iou", "detections_to_ground",
     "ground_to_detections", "ground_match_count", "label_match",
     "use_detection", "use_ground"))

def compare_labels(detections, ground_truth,
                   det_filter=None, annot_filter=None):
    """
    Given two sets of Label classes, determine how they match up with one
    another. Assumes both label class instances are sorted by start time.
    :param detections:  label object representing a set of detections
    :param ground_truth: label object representing "accurate" annotation of data
    :param det_filter:  Function used to filter a set of detections
       Must return a True/False array indicating that the detections should
       be used (True) or discarded (False) in scoring.
    :param annot_filter:  Function used to filter a set of annotations
       Must return a True/False array indicating that the annotations should
       be used (True) or discarded (False) in scoring.  Any detections that
       match these annotations will be treated as if they did not exist.
       That is, those detections will not be counted as matches or false
       positives.
    :return: namedtuple of ComparisonType.  Has fields:
       detections_iou - For each detection, show intersection over union (IOU)
                        with matched ground truth call.  Unmatched calls have
                        zero IOU.
       ground_iou - For each ground truth, IOU with matched detection
       detections_to_ground - For each detection, index of match in ground truth
                              or -1 if no match. If multiple matches are
                              possible, only one will be shown
       ground_to_detections - For each ground truth, index of match in
                              detections or -1 if no match. If multiple matches
                              are possible, only one will be shown
       label_match - For each detection, boolean indicating whether or not the
                      class name matches the class of any associated ground
                      truth
       use_detection - For each detection, boolean indicating whether or not it
                     should be counted.  When False, the detection matches
                     something in the annotation data that we were not expecting
                     to match and we do not want it to count as a false positive
       use_ground - For each ground-truth annotations, boolean indicating
                     whether or not
    """

    # count of labels in each set
    detectionsN = len(detections)
    ground_truthN = len(ground_truth)

    # Measure of intersection over union (IOU) for each call, a 0 indicates no match
    detections_iou = np.zeros((detectionsN,))
    ground_iou = np.zeros((ground_truthN,))
    # Indicate indices of matching call, -1 indicates no match
    detections_to_ground = np.ones((detectionsN,), dtype=np.int) * -1
    ground_to_detections = np.ones((ground_truthN,), dtype=np.int) * -1
    # It is possible to have multiple detections match a single ground truth.
    # We keep count of how many times this happens.  Note that under the current design, when this
    # does occur our IOU metrics will not really be correct as we will only track the last call.
    # If there end up being a lot of multiple matches, we should perhaps reconsider this design.
    ground_match_count = np.zeros((ground_truthN,), dtype=np.int)
    # Determines whether class label matches ground truth
    label_match = np.zeros((detectionsN,), dtype=np.bool)

    detectionsI = np.ones((detectionsN,), dtype=np.bool) if det_filter is None \
        else np.array(det_filter(detections), dtype=np.bool)
    groundI = np.ones((ground_truthN,), dtype=np.bool) if annot_filter is None \
        else np.array(annot_filter(ground_truth), dtype=np.bool)

    # For now, we will not worry about overlapping calls
    # Work our way through the lists, computing IOU as they arise
    d_idx = 0   # index into detections
    g_idx = 0  # index into ground truth
    done = d_idx >= detectionsN or g_idx > ground_truthN
    while not done:

        if detectionsI[d_idx] is False:
            d_idx += 1   # skip detection
        elif groundI[g_idx] is False:
            #todo:  How do we know if we matched a detection to this
            #todo:  Should possibly match but not remove
            g_idx += 1  # skip annotation
        else:
            """
            Checking overlap - multiple cases
            
            detection:                 s*********************e
            ground truth:      s:::e s-----e  s-------e   s------e  s::::e
                                   s---------------------------------e
            Ground truth calls with hyphens between start and end overlap.
            In all of the overlapped cases, the latest start time is before 
            the earliest end.
            """

            gt_start = ground_truth.data.iloc[g_idx].Start
            gt_end = ground_truth.data.iloc[g_idx].End
            d_start = detections.data.iloc[d_idx].Start
            d_end = detections.data.iloc[d_idx].End

            latest_start = max(d_start, gt_start)
            earliest_end = min(d_end, gt_end)
            overlap = earliest_end - latest_start
            # Originally had overlap > 0 as opposed to >= 0
            # but a bad ground truth label with zero duration resulted in
            # an infinite loop.
            overlapped = overlap.total_seconds() >= 0
            if overlapped:
                # Find union interval
                union_start = min(gt_start, d_start)
                union_end = max(gt_end, d_end)
                # Find intersection interval
                int_start = max(gt_start, d_start)
                int_end = min(gt_end, d_end)
                # intersection over union
                iou = (int_end - int_start) / (union_end - union_start)
                # record overlap and information to match up label sets
                detections_iou[d_idx] = iou
                ground_iou[g_idx] = iou
                detections_to_ground[d_idx] = g_idx
                ground_to_detections[g_idx] = d_idx

                # Is the class label correct?
                label_match[d_idx] = detections.data.iloc[d_idx].Class == ground_truth.data.iloc[g_idx].Name
                # Track counts of how many times the ground truth call was matched
                ground_match_count[g_idx] += 1  # IOU for this ground truth reflects IOU of the last match

                # Move to next detection.  Do not advance ground truth as a call may be detected in multiple parts
                d_idx += 1
            else:
                # Either the detection or ground truth is earlier, advance
                # the earlier one.
                if gt_end <= d_start:
                    # Ground truth before current detection
                    g_idx += 1
                elif gt_start >= d_end:
                    # Ground truth after current detection
                    d_idx += 1

        done = d_idx >= detectionsN or g_idx >= ground_truthN

    # Any place that that a ground truth annotation matches a detection,
    # change the detection so that it is not counted
    tmp = ground_to_detections[~groundI]
    for idx in tmp[tmp != -1]:
        detectionsI[idx] = False


    return ComparisonType(detections_iou, ground_iou,
                          detections_to_ground, ground_to_detections,
                          ground_match_count, label_match,
                          detectionsI, groundI)