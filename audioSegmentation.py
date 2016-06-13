import numpy
import audioBasicIO
import audioTrainTest as aT
import audioFeatureExtraction as aF
from sklearn.lda import LDA
from scipy.spatial import distance
import sklearn
import sklearn.hmm
import mlpy


def speakerDiarization(fileName, numOfSpeakers, mtSize=2.0, mtStep=0.2, stWin=0.05, LDAdim=35, PLOT=False):
    '''
    ARGUMENTS:
        - fileName:        the name of the WAV file to be analyzed
        - numOfSpeakers    the number of speakers (clusters) in the recording (<=0 for unknown)
        - mtSize (opt)     mid-term window size
        - mtStep (opt)     mid-term window step
        - stWin  (opt)     short-term window size
        - LDAdim (opt)     LDA dimension (0 for no LDA)
        - PLOT     (opt)   0 for not plotting the results 1 for plottingy
    '''
    [Fs, x] = audioBasicIO.readAudioFile(fileName)
    x = audioBasicIO.stereo2mono(x)
    Duration = len(x) / Fs

    [Classifier1, MEAN1, STD1, classNames1, mtWin1, mtStep1, stWin1, stStep1, computeBEAT1] = aT.loadKNNModel(
        "data/knnSpeakerAll")
    [Classifier2, MEAN2, STD2, classNames2, mtWin2, mtStep2, stWin2, stStep2, computeBEAT2] = aT.loadKNNModel(
        "data/knnSpeakerFemaleMale")

    [MidTermFeatures, ShortTermFeatures] = aF.mtFeatureExtraction(x, Fs, mtSize * Fs, mtStep * Fs, round(Fs * stWin),
                                                                  round(Fs * stWin * 0.5))

    MidTermFeatures2 = numpy.zeros(
        (MidTermFeatures.shape[0] + len(classNames1) + len(classNames2), MidTermFeatures.shape[1]))

    for i in range(MidTermFeatures.shape[1]):
        curF1 = (MidTermFeatures[:, i] - MEAN1) / STD1
        curF2 = (MidTermFeatures[:, i] - MEAN2) / STD2
        [Result, P1] = aT.classifierWrapper(Classifier1, "knn", curF1)
        [Result, P2] = aT.classifierWrapper(Classifier2, "knn", curF2)
        MidTermFeatures2[0:MidTermFeatures.shape[0], i] = MidTermFeatures[:, i]
        MidTermFeatures2[MidTermFeatures.shape[0]:MidTermFeatures.shape[0] + len(classNames1), i] = P1 + 0.0001
        MidTermFeatures2[MidTermFeatures.shape[0] + len(classNames1)::, i] = P2 + 0.0001

    MidTermFeatures = MidTermFeatures2  # TODO
    # SELECT FEATURES:
    # iFeaturesSelect = [8,9,10,11,12,13,14,15,16,17,18,19,20];                                                                                         # SET 0A
    # iFeaturesSelect = [8,9,10,11,12,13,14,15,16,17,18,19,20, 99,100];                                                                                 # SET 0B
    # iFeaturesSelect = [8,9,10,11,12,13,14,15,16,17,18,19,20, 68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,
    #   97,98, 99,100];     # SET 0C

    iFeaturesSelect = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                       53]  # SET 1A
    # iFeaturesSelect = [8,9,10,11,12,13,14,15,16,17,18,19,20,41,42,43,44,45,46,47,48,49,50,51,52,53, 99,100];                                          # SET 1B
    # iFeaturesSelect = [8,9,10,11,12,13,14,15,16,17,18,19,20,41,42,43,44,45,46,47,48,49,50,51,52,53, 68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98, 99,100];     # SET 1C

    # iFeaturesSelect = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53];             # SET 2A
    # iFeaturesSelect = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53, 99,100];     # SET 2B
    # iFeaturesSelect = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53, 68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98, 99,100];     # SET 2C

    # iFeaturesSelect = range(100);                                                                                                    # SET 3
    # MidTermFeatures += numpy.random.rand(MidTermFeatures.shape[0], MidTermFeatures.shape[1]) * 0.000000010

    MidTermFeatures = MidTermFeatures[iFeaturesSelect, :]

    (MidTermFeaturesNorm, MEAN, STD) = aT.normalizeFeatures([MidTermFeatures.T])
    MidTermFeaturesNorm = MidTermFeaturesNorm[0].T
    numOfWindows = MidTermFeatures.shape[1]

    # remove outliers:
    DistancesAll = numpy.sum(distance.squareform(distance.pdist(MidTermFeaturesNorm.T)), axis=0)
    MDistancesAll = numpy.mean(DistancesAll)
    iNonOutLiers = numpy.nonzero(DistancesAll < 1.2 * MDistancesAll)[0]

    # TODO: Combine energy threshold for outlier removal:
    # EnergyMin = numpy.min(MidTermFeatures[1,:])
    # EnergyMean = numpy.mean(MidTermFeatures[1,:])
    # Thres = (1.5*EnergyMin + 0.5*EnergyMean) / 2.0
    # iNonOutLiers = numpy.nonzero(MidTermFeatures[1,:] > Thres)[0]
    # print iNonOutLiers

    perOutLier = (100.0 * (numOfWindows - iNonOutLiers.shape[0])) / numOfWindows
    MidTermFeaturesNormOr = MidTermFeaturesNorm
    MidTermFeaturesNorm = MidTermFeaturesNorm[:, iNonOutLiers]

    # LDA dimensionality reduction:
    if LDAdim > 0:
        # [mtFeaturesToReduce, _] = aF.mtFeatureExtraction(x, Fs, mtSize * Fs, stWin * Fs, round(Fs*stWin), round(Fs*stWin));
        # extract mid-term features with minimum step:
        mtWinRatio = int(round(mtSize / stWin))
        mtStepRatio = int(round(stWin / stWin))
        mtFeaturesToReduce = []
        numOfFeatures = len(ShortTermFeatures)
        numOfStatistics = 2
        # for i in range(numOfStatistics * numOfFeatures + 1):
        for i in range(numOfStatistics * numOfFeatures):
            mtFeaturesToReduce.append([])

        for i in range(numOfFeatures):  # for each of the short-term features:
            curPos = 0
            N = len(ShortTermFeatures[i])
            while (curPos < N):
                N1 = curPos
                N2 = curPos + mtWinRatio
                if N2 > N:
                    N2 = N
                curStFeatures = ShortTermFeatures[i][N1:N2]
                mtFeaturesToReduce[i].append(numpy.mean(curStFeatures))
                mtFeaturesToReduce[i + numOfFeatures].append(numpy.std(curStFeatures))
                curPos += mtStepRatio
        mtFeaturesToReduce = numpy.array(mtFeaturesToReduce)
        mtFeaturesToReduce2 = numpy.zeros(
            (mtFeaturesToReduce.shape[0] + len(classNames1) + len(classNames2), mtFeaturesToReduce.shape[1]))
        for i in range(mtFeaturesToReduce.shape[1]):
            curF1 = (mtFeaturesToReduce[:, i] - MEAN1) / STD1
            curF2 = (mtFeaturesToReduce[:, i] - MEAN2) / STD2
            [Result, P1] = aT.classifierWrapper(Classifier1, "knn", curF1)
            [Result, P2] = aT.classifierWrapper(Classifier2, "knn", curF2)
            mtFeaturesToReduce2[0:mtFeaturesToReduce.shape[0], i] = mtFeaturesToReduce[:, i]
            mtFeaturesToReduce2[mtFeaturesToReduce.shape[0]:mtFeaturesToReduce.shape[0] + len(classNames1),
            i] = P1 + 0.0001
            mtFeaturesToReduce2[mtFeaturesToReduce.shape[0] + len(classNames1)::, i] = P2 + 0.0001
        mtFeaturesToReduce = mtFeaturesToReduce2
        mtFeaturesToReduce = mtFeaturesToReduce[iFeaturesSelect, :]
        # mtFeaturesToReduce += numpy.random.rand(mtFeaturesToReduce.shape[0], mtFeaturesToReduce.shape[1]) * 0.0000010
        (mtFeaturesToReduce, MEAN, STD) = aT.normalizeFeatures([mtFeaturesToReduce.T])
        mtFeaturesToReduce = mtFeaturesToReduce[0].T
        # DistancesAll = numpy.sum(distance.squareform(distance.pdist(mtFeaturesToReduce.T)), axis=0)
        # MDistancesAll = numpy.mean(DistancesAll)
        # iNonOutLiers2 = numpy.nonzero(DistancesAll < 3.0*MDistancesAll)[0]
        # mtFeaturesToReduce = mtFeaturesToReduce[:, iNonOutLiers2]
        Labels = numpy.zeros((mtFeaturesToReduce.shape[1],))
        LDAstep = 1.0
        LDAstepRatio = LDAstep / stWin
        # print LDAstep, LDAstepRatio
        for i in range(Labels.shape[0]):
            Labels[i] = int(i * stWin / LDAstepRatio);
        clf = LDA(n_components=LDAdim)
        clf.fit(mtFeaturesToReduce.T, Labels, tol=0.000001)
        MidTermFeaturesNorm = (clf.transform(MidTermFeaturesNorm.T)).T

    if numOfSpeakers <= 0:
        sRange = range(2, 10)
    else:
        sRange = [numOfSpeakers]
    clsAll = []
    silAll = []
    centersAll = []

    for iSpeakers in sRange:
        cls, means, steps = mlpy.kmeans(MidTermFeaturesNorm.T, k=iSpeakers, plus=True)  # perform k-means clustering

        # YDist =   distance.pdist(MidTermFeaturesNorm.T, metric='euclidean')
        # print distance.squareform(YDist).shape
        # hc = mlpy.HCluster()
        # hc.linkage(YDist)
        # cls = hc.cut(14.5)
        # print cls

        # Y = distance.squareform(distance.pdist(MidTermFeaturesNorm.T))
        clsAll.append(cls)
        centersAll.append(means)
        silA = [];
        silB = []
        for c in range(iSpeakers):  # for each speaker (i.e. for each extracted cluster)
            clusterPerCent = numpy.nonzero(cls == c)[0].shape[0] / float(len(cls))
            if clusterPerCent < 0.020:
                silA.append(0.0)
                silB.append(0.0)
            else:
                MidTermFeaturesNormTemp = MidTermFeaturesNorm[:, cls == c]  # get subset of feature vectors
                Yt = distance.pdist(
                    MidTermFeaturesNormTemp.T)  # compute average distance between samples that belong to the cluster (a values)
                silA.append(numpy.mean(Yt) * clusterPerCent)
                silBs = []
                for c2 in range(iSpeakers):  # compute distances from samples of other clusters
                    if c2 != c:
                        clusterPerCent2 = numpy.nonzero(cls == c2)[0].shape[0] / float(len(cls))
                        MidTermFeaturesNormTemp2 = MidTermFeaturesNorm[:, cls == c2]
                        Yt = distance.cdist(MidTermFeaturesNormTemp.T, MidTermFeaturesNormTemp2.T)
                        silBs.append(numpy.mean(Yt) * (clusterPerCent + clusterPerCent2) / 2.0)
                silBs = numpy.array(silBs)
                silB.append(min(silBs))  # ... and keep the minimum value (i.e. the distance from the "nearest" cluster)
        silA = numpy.array(silA)
        silB = numpy.array(silB)
        sil = []
        for c in range(iSpeakers):  # for each cluster (speaker)
            sil.append((silB[c] - silA[c]) / (max(silB[c], silA[c]) + 0.00001))  # compute silhouette

        silAll.append(numpy.mean(sil))  # keep the AVERAGE SILLOUETTE

    # silAll = silAll * (1.0/(numpy.power(numpy.array(sRange),0.5)))
    imax = numpy.argmax(silAll)  # position of the maximum sillouette value
    nSpeakersFinal = sRange[imax]  # optimal number of clusters

    return nSpeakersFinal
