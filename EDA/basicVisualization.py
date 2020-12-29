import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os
import datetime

def categorizeLabel(x):
    '''
    Classify the label to `Normal` and `Botnet`
    '''
    if 'Botnet' in x:
        return 'Botnet'
    return 'Normal'

def clean(data):
    '''
    Simple preprocessing/ cleaning
    '''
    data['StartTime'] = pd.to_datetime(data['StartTime'])
    data['Label'] = data['Label'].apply(categorizeLabel)

def pickupFig1(data, prefixFigurePath='Figures'):
    '''
    Frequency of flow with respect to time
    '''
    data.resample('T', on='StartTime').count()['StartTime'].plot()
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.title('Figure 1. Frequency of flow w.r.t time')
    plt.savefig(f'{prefixFigurePath}/Fig1.png')

def pickupFig2(data, prefixFigurePath='Figures'):
    '''
    Frequency of flow with respect to duration
    '''
    countDur = data['Dur'].apply(lambda x: int(x))
    sns.distplot(countDur, kde=False)
    plt.xlabel('Duration (in seconds)')
    plt.ylabel('Frequency')
    plt.title('Figure 2. Frequency of flow w.r.t duration')
    plt.savefig(f'{prefixFigurePath}/Fig2.png')

def pickupFig3(data, prefixFigurePath='Figures'):
    '''
    Distribution of labels
    '''
    sns.countplot(x='Label', data=data)
    plt.title('Figure 3. Distribution of labels')
    plt.savefig(f'{prefixFigurePath}/Fig3.png')

def pickupFig4(data, prefixFigurePath='Figures'):
    '''
    Frequency of flow with respect to total of packets
    '''
    data.resample('T', on='StartTime').sum()['TotPkts'].plot()
    plt.xlabel('Date')
    plt.ylabel('Total of packets')
    plt.title('Figure 4. Frequency of flow w.r.t total of packets')
    plt.savefig(f'{prefixFigurePath}/Fig4.png')

def pickupFig5(data, prefixFigurePath='Figures'):
    '''
    Frequency of flow with respect to total of packets between 12:10 and 12:28
    '''
    # find out the max total of packer in five minutes sample
    fiveMinsSample = data.resample('5T', on='StartTime').sum()['TotPkts']
    target = fiveMinsSample[fiveMinsSample == fiveMinsSample.max()].index
    targetDatetime = target.to_pydatetime()[0]

    # find -/+ 10 minunite from the point of max total packet
    lower = targetDatetime - datetime.timedelta(minutes=10)
    upper = targetDatetime + datetime.timedelta(minutes=10)

    # plot out the range of point which has max total packet
    attacked = data[(data['StartTime'] > lower) & (data['StartTime'] < upper)]
    attacked.resample('T', on='StartTime').sum()['TotPkts'].plot()
    plt.xlabel('Date')
    plt.ylabel('Total of packets')
    plt.title('Freq of flow w.r.t total of packets in time of attack')
    plt.savefig(f'{prefixFigurePath}/Fig5.png')
    

if __name__ == '__main__':
    parentPath = '.'

    # declare the path of Netflow file
    prefixPath = f'{parentPath}/CTU-13-Dataset'
    scenarioFile=f'{parentPath}/{prefixPath}/capture20110810.binetflow'

    # declare the prefix path of figures
    prefixFigurePath = f'{parentPath}/Figures'
    if not os.path.exists(prefixFigurePath):
        os.mkdir(prefixFigurePath)

    # load the train data
    trainDF = pd.read_csv(scenarioFile)

    # clean DB
    clean(trainDF)

    # Visualization
    pickupFig1(trainDF)
    pickupFig2(trainDF)
    pickupFig3(trainDF)
    pickupFig4(trainDF)
    pickupFig5(trainDF)