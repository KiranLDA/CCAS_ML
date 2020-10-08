import numpy as np
import matplotlib.pyplot as plt

def get_window_call_range(frame):
    ''' For every window, determine the start and end timesteps of continuous calls in the label'''
    indices = []
    arr = frame[:, :8].T
    for i in range(len(arr)):
        call_indices = np.where(arr[i] != 0)
        sequences = np.split(call_indices[0], np.where(np.diff(call_indices[0]) > 1)[0] + 1)
        timesteps = []
        for s in sequences:
            if len(s) > 0: timesteps.append([s[0], s[len(s) - 1]])
        indices.append(timesteps)
    return indices


def get_call_ranges(y):
    call_ranges = []
    for frame in y:
        indices = get_window_call_range(frame)
        call_ranges.append(indices)
    return np.asarray(call_ranges)


def get_fragments(y_gt, y_pred, indices):
    '''Returns 2 matrices, each of shape [num_samples*num_calls]
     The first matrix returns a 2d array which represents the silence timesteps across multiple continuous calls, for a sample wrt to a label(call-type).
     Example: Consider a sample (6 sec frame) has call between [[65, 153]] for label 0 (i.e. GIG) and the network predicted call between [[66,127],[130,133],[136,136],[140,151]], thus the value for silence_details[sample][0] = [128, 129, 134, 135, 137, 138, 139] since the network detected no call for the duration
     The second matrix returns a 2d array which represents the total number of fragments across different calls for a particular sample wrt to label
     In the above example, the fragment for the 6sec frame against label 0 is '4' since the call was detected across 4 fragments after thresholding with the OTH label
     '''

    fragment_details = np.zeros((y_gt.shape[0], 8))
    silence_details = np.zeros((y_gt.shape[0], 8), dtype=np.ndarray)

    for sample, (pred, sample_call_ranges) in enumerate(zip(y_pred, indices)):
        for label_idx, label_ranges in enumerate(sample_call_ranges):
            for call_range in label_ranges:
                silence = get_silence_duration(call_range, pred, label_idx)
                if len(silence) > 0:
                    if type(silence_details[sample][label_idx]) is int:
                        silence_details[sample][label_idx] = silence
                    else:
                        silence_details[sample][label_idx] = np.concatenate([silence_details[sample][label_idx], silence])
                    # Checks the difference between subsequent silence timesteps, to detect contiguous silence time durations
                    # Example silence =[45,46,47, 65,66,67,68], there fragments = 3 because it will check the continuous difference values, here [1,1, 18, 1, 1, 1] + "2" because every silence chunk divides into two fragments
                    fragments = len(np.where(np.diff(silence) > 1)[0]) + 2
                    fragment_details[sample][label_idx] = fragments
    return [silence_details, fragment_details]


def get_silence_duration(call_range, pred, idx):
        start = call_range[0]
        end = call_range[1]
        silence = np.where(np.trim_zeros(pred[:, idx][start:end]) == 0)
        silence = list(map(lambda x: x + get_start_delta(pred, idx, start, end), silence[0]))
        return silence

def get_start_delta(pred, idx, start, end):
    delta = start
    for val in pred[:, idx][start: end]:
        if val != 0: break
        delta += 1
    return delta

def plot_fragments(fragments,label):
    samples = fragments.shape[0]
    fragments[fragments==0] = np.nan
    plt.scatter(np.arange(samples), fragments)
    plt.xticks(np.arange(samples, step=100))
    plt.title("Fragment Count for label: "+label)
    plt.savefig('Fragement_' + label + '.png')
    plt.show()
    plt.close()