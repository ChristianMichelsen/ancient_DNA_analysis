import numpy as np
from tqdm import tqdm
import pysam
import extra_funcs
from Bio.Seq import Seq


paths, cfg = extra_funcs.load_yamls()


def load_data():

    neand = pysam.AlignmentFile(paths["bam_neand"], "rb")
    modern = pysam.AlignmentFile(paths["bam_modern"], "rb")

    neand_seqs = []
    for j in range(1, 2):
        chromosone = "chr" + str(j)
        print(chromosone)
        it = neand.fetch(chromosone)
        for i in it:
            if i.infer_query_length() == 76:
                s = str(i.get_forward_sequence())
                if (
                    s.count("A") > 0
                    and s.count("C") > 0
                    and s.count("G") > 0
                    and s.count("T") > 0
                    and "N" not in s
                ):
                    neand_seqs.append(i.get_forward_sequence())
    len(neand_seqs)

    neand_reverse_complement_seqs = [str(Seq(i).reverse_complement()) for i in neand_seqs]
    neand_reverse_complement_seqs[0:10]

    neand_seqs_augmented = neand_seqs + neand_reverse_complement_seqs
    len(neand_seqs_augmented)

    modern_seqs = []
    for j in range(1, 2):
        chromosone = "chr" + str(j)
        print(chromosone)
        it = modern.fetch(chromosone)
        for i in it:
            if len(modern_seqs) == len(neand_seqs):
                break
            else:
                s = str(i.get_forward_sequence())
                if (
                    s.count("A") > 0
                    and s.count("C") > 0
                    and s.count("G") > 0
                    and s.count("T") > 0
                    and "N" not in s
                ):
                    modern_seqs.append(i.get_forward_sequence())
    len(modern_seqs)

    modern_reverse_complement_seqs = [str(Seq(i).reverse_complement()) for i in modern_seqs]

    modern_seqs_augmented = modern_seqs + modern_reverse_complement_seqs
    len(modern_seqs_augmented)

    sequences = neand_seqs_augmented + modern_seqs_augmented
    len(sequences)

    labels = list(np.ones(len(neand_seqs_augmented))) + list(np.zeros(len(modern_seqs_augmented)))
    len(labels)

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    # The LabelEncoder encodes a sequence of bases as a sequence of integers: 0, 1, 2 and 3
    integer_encoder = LabelEncoder()
    # The OneHotEncoder converts an array of integers to a sparse matrix where
    # each row corresponds to one possible value of each feature, i.e. only 01 and 1 are present in the matrix
    one_hot_encoder = OneHotEncoder()
    input_features = []

    for sequence in tqdm(sequences):
        integer_encoded = integer_encoder.fit_transform(list(sequence))
        integer_encoded = np.array(integer_encoded).reshape(-1, 1)
        one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
        input_features.append(one_hot_encoded.toarray().T)

    np.set_printoptions(threshold=40)
    # print(input_features.shape)
    input_features = np.stack(input_features)
    print("Example sequence\n-----------------------")
    print("DNA Sequence #1:\n", sequences[0][:10], "...", sequences[0][-10:])
    print("One hot encoding of Sequence #1:\n", input_features[0].T)

    one_hot_encoder = OneHotEncoder()
    labels = np.array(labels).reshape(-1, 1)
    input_labels = one_hot_encoder.fit_transform(labels).toarray()

    print("Labels:\n", labels.T)
    print("One-hot encoded labels:\n", input_labels.T)

    return input_features, input_labels


# if False:

#     from data import comparison

#     input_features, input_labels = comparison.load_data()

#     from sklearn.model_selection import train_test_split

#     train_features, test_features, train_labels, test_labels = train_test_split(
#         input_features, input_labels, test_size=0.8, random_state=42
#     )

#     X_train, X_valid = torch.tensor(train_features).float(), torch.tensor(test_features).float()
#     y_train, y_valid = torch.tensor(train_labels).float(), torch.tensor(test_labels).float()

#     if cfg["use_gpu"]:
#         X_train = X_train.cuda()
#         X_valid = X_valid.cuda()
#         y_train = y_train.cuda()
#         y_valid = y_valid.cuda()

#     train_ds = TensorDataset(X_train, y_train)  # FIRST 1500 data points
#     valid_ds = TensorDataset(X_valid, y_valid)

