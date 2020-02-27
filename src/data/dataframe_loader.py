from pathlib import Path
import yaml
import pandas as pd
from timer import Timer
import bam_reader
import extra_funcs


paths, cfg = extra_funcs.load_yamls()


def _load_df_neand_and_modern(chromosones, num_cores):

    N_chromosones = len(chromosones)

    if cfg["force_rerun"] or not (
        Path(paths["df_neand"].format(N_chromosones=N_chromosones)).is_file()
        and Path(paths["df_modern"].format(N_chromosones=N_chromosones)).is_file()
    ):

        with Timer("Reading Neandertal BAM file"):
            seqs_neand = bam_reader.get_seqs_from_chromosones(
                paths["bam_neand"], chromosones, num_cores=num_cores
            )
        len(seqs_neand)

        with Timer("Reading Modern BAM file"):
            seqs_modern = bam_reader.get_seqs_from_chromosones(
                paths["bam_modern"], chromosones, num_cores=num_cores, max_reads=len(seqs_neand)
            )
        len(seqs_modern)

        df_seq_neand = pd.DataFrame(seqs_neand, columns=["sequences"])
        df_seq_neand["label"] = 1
        df_seq_neand.to_pickle(paths["df_neand"].format(N_chromosones=N_chromosones))

        df_seq_modern = pd.DataFrame(seqs_modern, columns=["sequences"])
        df_seq_modern["label"] = 0
        df_seq_modern.to_pickle(paths["df_modern"].format(N_chromosones=N_chromosones))

    else:

        df_seq_neand = pd.read_pickle(paths["df_neand"].format(N_chromosones=N_chromosones))
        df_seq_modern = pd.read_pickle(paths["df_modern"].format(N_chromosones=N_chromosones))

    return df_seq_neand, df_seq_modern


def _downsample_dataframe(df, N, method="random"):
    # TODO Note this downsampling step to ensure 50/50 mix of ancient / modern
    if method.lower() == "random":
        return df.sample(N, replace=False, random_state=42)
    elif method.lower() == "ordered":
        return df.iloc[:N].copy()
    else:
        assert False


def _join_neand_modern(df_seq_neand, df_seq_modern, N_chromosones, do_shuffle=True):
    if (
        cfg["force_rerun"]
        or not Path(paths["df_combined"].format(N_chromosones=N_chromosones)).is_file()
    ):
        df_seq_modern_downsampled = _downsample_dataframe(
            df_seq_modern, len(df_seq_neand), method="ordered"
        )

        df_seq = pd.concat([df_seq_neand, df_seq_modern_downsampled], ignore_index=True)
        if do_shuffle:
            df_seq = df_seq.sample(frac=1, replace=False, random_state=42)
        df_seq.to_pickle(paths["df_combined"].format(N_chromosones=N_chromosones))
    else:
        df_seq = pd.read_pickle(paths["df_combined"].format(N_chromosones=N_chromosones))
    return df_seq


def load_seqs(chromosones, num_cores=-1, do_shuffle=True):
    N_chromosones = len(chromosones)
    df_seq_neand, df_seq_modern = _load_df_neand_and_modern(chromosones, num_cores=num_cores)
    df_seq = _join_neand_modern(df_seq_neand, df_seq_modern, N_chromosones, do_shuffle)
    return df_seq
