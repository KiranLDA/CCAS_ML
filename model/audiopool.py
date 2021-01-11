
import os
import numpy as np

# Add-on modules, use conda or pip to install
import soundfile
from pylru import lrucache  # least recently used cache, note functools.lru_cache
                            # is an alternative after Python 3.8 (problems
                            # with releasing object prior to this), would
                            # require reimplementation of cache system below

class AudioPool:

    def __init__(self, size=512):
        """
        AudioPool maintains a least recently used cache of open files.
        Files are accessed
        If the file is opened and in the pool,
        """

        # Create least recently used cache of specified size
        self.cache = lrucache(size)

    def __getitem__(self, filename):
        """
        [filename] - Retrieve Soundfile instance for the specified filename
        """
        if filename in self.cache:
            sound = self.cache[filename]
        else:
            # Not in cache
            sound = soundfile.SoundFile(filename)
            self.cache[filename] = sound

        return sound

    def get_samples(self, filename, start_sample, Nsamples=-1):
        """
        get_samples(filename, start_sample, Nsamples)
        Read Nsamples of data (from all channnels) from file filename
        starting at sample start_sample.  If Nsamples < 0, data to the
        end of the file are returned.

        If start_sample is None, reads from current position
        """

        sound = self[filename]
        if start_sample is not None:
            # Ensure we are at the start position
            current = sound.tell()
            if current != start_sample:
                sound.seek(start_sample)  # Move to the desired position

        data = sound.read(frames=Nsamples)
        return data

    def get_seconds(self, filename, start_s, duration_s=-1):
        """
        get_seconds(filename, start_s, duration_s)
        Return duration_s seconds of data from filename starting at start_s
        seconds into the file.

        If start_s is None, reads from current position.
        if duration_s is -1, reads to end of file
        """

        sound = self[filename]
        Fs = sound._info.samplerate

        if start_s is not None:
            start_sample = int(start_s * Fs)
            # Ensure we are at the correct start position
            current = sound.tell()
            if current != start_sample:
                sound.seek(start_sample)  # Move to desired position

        if duration_s == -1:
            Nsamples = -1
        else:
            Nsamples = int(Fs * duration_s)

        data = sound.read(frames=Nsamples)
        return data

    def get_metadata(self, filename):
        "Return metadata (e.g. channels, samplerate, encoding) for file"
        sound = self[filename]
        return sound._info

    def get_Fs(self, filename):
        "Return samplerate for file"
        sound = self[filename]
        return sound._info.samplerate


if __name__ == "__main__":
    # Run tests
    pool = AudioPool()  # Create audio pool

    # fname = "C:\hdd\corpora\CCAS\data\meerkat_data\meerkat_data_2017\HM_2017\HM_20170803\HM_VLF206_SOUNDFOC_20170803\HM_VLF206_SOUNDFOC_20170803_2.WAV"
    rootdir = "C:\hdd\corpora\CCAS\data\meerkat_data\meerkat_data_2017\HM_2017_2\COLLAR2\AUDIO2\HM_HTB_R14_20170821-20170825"
    fname = os.path.join(rootdir, "HM_HTB_R14_file_1_(2017_08_20-06_44_59)_ASWMUX221052.wav")
    fname2 = os.path.join(rootdir, "HM_HTB_R14_file_6_(2017_08_25-06_44_59)_ASWMUX221052.wav")

    # Retrieve metadata
    print(f'Sample rate {pool.get_Fs(fname)}')
    print("Metadata")
    meta = pool.get_metadata(fname)
    print(meta)

    # Get samples
    data = pool.get_samples(fname, 6000, 16)
    print(data)

    # Show getting the same data with sample indices or time
    Fs = pool.get_Fs(fname2)
    start_s = 25  # Read N s into file
    duration_s = 0.02  # 20 ms
    start_sample = int(start_s * Fs)
    duration_N = int(duration_s * Fs)
    # Two ways of getting the same data
    # Note that due to rounding differences, you may sometimes be one
    # sample different than what you expected.  In this example, we use
    # the same rounding rules as the code and there should not be any difference
    data = pool.get_samples(fname2, start_sample, duration_N)
    print(f"Start of data from get_samples({os.path.basename(fname2)}({start_sample}, {duration_N})")
    N = 20
    print(data[0:N])
    samedata = pool.get_seconds(fname2, start_s, duration_s)
    print(f"Start of data from get_seconds({os.path.basename(fname2)}({start_s}, {duration_s})")
    print(samedata[0:N])

    # Check we read the same thing
    results = np.where(data != samedata)
    if results[0].size > 0:
        print("Data between get_samples and get_seconds varied at indices:")
        print(results[0])

    print('done')





