import pysam
import numpy as np
import multiprocessing
from functools import partial
import itertools
from Bio.Seq import Seq
import re


def substitute_unwanted_charectors(s):
    "replace everything thats not in acgtn with X"
    return re.sub("[^ACGTN]", "X", s.upper())


def process_read(read):
    if read.infer_query_length() == 76:
        seq = str(read.get_forward_sequence())
        if (
            seq.count("A") > 0
            and seq.count("C") > 0
            and seq.count("G") > 0
            and seq.count("T") > 0
            and "N" not in seq
        ):
            return substitute_unwanted_charectors(seq)


def get_seq_from_single_chromosone(chromosone, alignment_filename, max_reads=np.inf):
    seqs = []
    alignment_file = pysam.AlignmentFile(alignment_filename, "rb")
    reads = alignment_file.fetch(chromosone, multiple_iterators=True)
    for read in reads:

        # break out of loop if length of seqs longer than max length
        if len(seqs) > max_reads / 2:  # factor 2 due to augmented reads
            break

        if read.infer_query_length() == 76:
            seq = process_read(read)
            if seq:
                seqs.append(seq)

    reverse_complement_seqs = [str(Seq(i).reverse_complement()) for i in seqs]
    seqs_augmented = seqs + reverse_complement_seqs
    return seqs_augmented


def get_seqs_from_chromosones(alignment_filename, chromosones, num_cores=-1, max_reads=np.inf):

    """ max_reads: max number of reads to extract pr. chromosone! Conservative measure to avoid trying to extract more reads from a single chromosone than available.
    """

    if num_cores == -1:
        num_cores = multiprocessing.cpu_count() - 1

    workers = multiprocessing.Pool(num_cores)
    seqs_all_chromosones = workers.map_async(
        partial(
            get_seq_from_single_chromosone,
            alignment_filename=alignment_filename,
            max_reads=max_reads,
        ),
        chromosones,
    ).get(9999999)

    # seqs_all_chromosones_flattened = [item for sublist in seqs_all_chromosones for item in sublist]
    seqs_all_chromosones_flattened = list(itertools.chain.from_iterable(seqs_all_chromosones))

    return seqs_all_chromosones_flattened
