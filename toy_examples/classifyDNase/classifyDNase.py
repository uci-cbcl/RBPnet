import pybedtools
from pyfasta import Fasta
from optparse import OptionParser
import numpy as np

from keras.models import Sequential
# added missing imports
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense

np.random.seed(1337)#random seed for reproducibility

def getOptions():
    parser = OptionParser()
    parser.add_option("--bed", dest = "bed", help = "Input bed file",
                      metavar = "FILE", type = "string", default = "")
    parser.add_option("--fa", dest = "fasta",
                      help = "reference genome in .fa format", metavar = "FILE", type = "string", default = "")
    (options, args) = parser.parse_args()
    return options

def main():
    options = getOptions()

    bed = options.bed
    genome = options.fasta

    peaks = pybedtools.BedTool(bed)
    rand = peaks.shuffle(genome='hg19', chrom = True, noOverlapping = True, seed = 1)
    fa = Fasta(genome)    

    # Put sequences into list
    peakSeqs = getAllSeqs(peaks, fa)
    randSeqs = getAllSeqs(rand, fa)
    n = len(peakSeqs)
    
    # Convert seqs to one hot
    peakSeqs.extend(randSeqs)
    x_train = seqsToOneHot(peakSeqs)
    list_labels = ([1]*n)
    list_labels.extend([0]*n)     
    y_train = np.array(list_labels)

    # Sequence classification with cnn
    wid = max(len(w) for w in peakSeqs)
    model = Sequential() 
    # increased motif size from 3 to 30. input_dim was incorrectly set to (4, wid)
    model.add(Convolution1D(64, 30, border_mode='valid', input_length=wid, input_dim=4))
    model.add(MaxPooling1D(pool_length = 15))
    model.add(Dense(100, activation = 'relu'))
    # added Flatten layer
    model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid'))
    # added accuracy metric
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # set shuffle to True
    model.fit(x_train, y_train, nb_epoch = 5, batch_size = 30, shuffle=True)
 

def getAllSeqs(bedRegions, fa):
    seqs = []
    for b in bedRegions:
        chrom = b.chrom 
        start = b.start
        end = b.stop
        currSeq = fa[chrom][start:end].upper()
        seqs.append(currSeq)
    return seqs

def seqsToOneHot(seqs):
    # Convert sequences to one hot
    wid = max(len(w) for w in seqs)
    a = np.array([ list(w.center(wid)) for w in seqs])
    
    n = np.array(['A', 'C', 'G', 'T'])
    b = a[:, :, np.newaxis] == n

    return b
 
main()

