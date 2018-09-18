
# Author: Ruben Glatt
# Modified by Felipe Leno
# This code countains functions to open .csv files and print graphs. Adaptation of the code published in:
#  Silva et al. Simultaneously Learning and Advising in Multiagent Reinforcement Learning. AAMAS-2017.
#This is an auxiliary source to be used together with the jupyter notebook file as explained in the README file.
#
import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
from collections import defaultdict
#sns.set(style="darkgrid")

import scipy as sp
import scipy.stats

def collect_experiment_data(source='/',algorithm="", runs=2):
    # load all agent data
    evalReward = defaultdict(list)
    evalWin = defaultdict(list)
    evalDraw = defaultdict(list)
    evalLose = defaultdict(list)
    evalTrials = np.array([])


    goodRuns = 0
    for run in range(0, runs):
        evalFile = os.path.join(source, algorithm, 'discRew' + str(run) + ".csv")
            
        #print evalFile
        if os.path.isfile(evalFile):
            try:
                _etime, _ereward, _ewin, _edraw, _elose = np.loadtxt(open(evalFile, "rb"), skiprows=1, delimiter=",", unpack=True)
            except:
                continue
            if sum(evalTrials)==0:
                evalTrials = _etime
                #
            if sum(_etime.shape) == sum(evalTrials.shape):
                goodRuns += 1
                for i in range(len(_etime)):
                    evalReward[_etime[i]].append(_ereward[i])
                    evalWin[_etime[i]].append(_ewin[i])
                    evalDraw[_etime[i]].append(_edraw[i])
                    evalLose[_etime[i]].append(_elose[i])
            #else:
                #print("Error " + str(run+1) + " - "+ str(sum(_etime.shape))+" , "+str(sum(evalTrials.shape)))
    #with open(os.path.join(source, "_"+ str(0) +"_"+ str(1) +"_AGENT_"+ str(1) +"_RESULTS_eval"), 'r') as f:
    #    first_line = f.readline()    
    #if first_line.split(',')[0]=='steps':
    #    episodes = False
    #else:
    #    episodes = True
    
    print('Could use %d runs from expected %d' % (goodRuns, runs)) 
 
    #print('len(evalGoalPercentages) %d --> %s %s' % (len(evalGoalPercentages), str(type(evalGoalPercentages[(1,20)])), str(evalGoalPercentages[(1,20)]) ))
    #print('len(evalGoalTimes) %d --> %s %s' % (len(evalGoalTimes), str(type(evalGoalTimes[(1,20)])), str(evalGoalTimes[(1,20)]) ))



    headerLine = []
    headerLine.append("time")
    for run in range(1, runs+1):
        headerLine.append("Run"+str(run))

    with open(os.path.join(source, "__EVAL_reward"), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow((headerLine))
            csvfile.flush()
            for key in evalReward.keys():
                newrow = [key]
                for i in evalReward[key]:
                    newrow.append("{:.2f}".format(i))
                csvwriter.writerow((newrow))
                csvfile.flush()
    
    with open(os.path.join(source, "__EVAL_win"), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow((headerLine))
            csvfile.flush()
            for key in evalWin.keys():
                newrow = [key]
                for i in evalWin[key]:
                    newrow.append("{:.2f}".format(i))
                csvwriter.writerow((newrow))
                csvfile.flush()

    with open(os.path.join(source, "__EVAL_draw"), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow((headerLine))
            csvfile.flush()
            for key in evalDraw.keys():
                newrow = [key]
                for i in evalDraw[key]:
                    newrow.append("{:.2f}".format(i))
                csvwriter.writerow((newrow))
                csvfile.flush()

    with open(os.path.join(source, "__EVAL_lose"), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow((headerLine))
        csvfile.flush()
        for key in evalLose.keys():
            newrow = [key]
            for i in evalLose[key]:
                newrow.append("{:.2f}".format(i))
            csvwriter.writerow((newrow))
            csvfile.flush()
    

def summarize_data(data, confidence=0.95):
    n = len(data)
    m = np.nanmean(data,axis=1)
    import scipy.stats as stats
    se = stats.sem(data,axis=1,nan_policy='omit')
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return np.asarray([m, m-h, m+h])

def diff_avg_data(source1, source2, nameOutput="", confidence=0.95):
    values = ["__EVAL_reward"]
    for value in values:
        evalFile1 = os.path.join(source1, value)
        evalFile2 = os.path.join(source2, value)
        try:
            evalFileContent1 = pd.read_csv(open(evalFile1, "rb"), skiprows=0, delimiter=",")
            evalFileContent2 = pd.read_csv(open(evalFile2, "rb"), skiprows=0, delimiter=",")
        except Exception as e:
            print(e)
            
            
        
        values1 = evalFileContent1.values
        values2 = evalFileContent2.values
        import operator
        listToSort = np.asarray(values1).tolist()#.tolist()
        listToSort.sort(key=operator.itemgetter(0))
        values1 = np.array(listToSort)
        listToSort = np.asarray(values2).tolist()#.tolist()
        listToSort.sort(key=operator.itemgetter(0))
        values2 = np.array(listToSort)
        trials = values1[:,0]
        data = values1[:,1:] - values2[:,1:]
        
        update = summarize_data(data,confidence)
        headerLine = []
        

        headerLine.append("time")
        headerLine.append("mean")
        headerLine.append("ci_down")
        headerLine.append("ci_up")

        value = value.replace("EVAL","DIFF_"+nameOutput)
        with open(os.path.join(source1, value), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerow((headerLine))
            csvfile.flush()

            for i in range(sum(trials.shape)):
                newrow = [trials[i]]
                for j in update.T[i]:
                    newrow.append("{:.2f}".format(j))
                csvwriter.writerow((newrow))
                csvfile.flush()
    
    
    
def summarize_experiment_data(source, confidence=0.95):
    values = ["__EVAL_reward", "__EVAL_win","__EVAL_draw", "__EVAL_lose"]
    
    for value in values:
        evalFile = os.path.join(source, value)
        #print(evalFile)
        #evalFileContent = np.loadtxt(open(evalFile, "rb"), skiprows=1, delimiter=",", unpack=True)
        try:
            evalFileContent = pd.read_csv(open(evalFile, "rb"), skiprows=0, delimiter=",")
        except Exception as e:
            print("Problem in File: "+evalFile)
            print(e)
            
        
        values = evalFileContent.values
        #------------------------------------------------------------------------------------------------------------------
        #Evaluates the data and removes episodes when they have less than 20% of points when compared to the maximum found
        # ------------------------------------------------------------------------------------------------------------------
        values = filter_small_samples(values, 0.2)
        import operator
        listToSort = np.asarray(values).tolist()#.tolist()
        listToSort.sort(key=operator.itemgetter(0))
        values = np.array(listToSort)
        trials = values[:,0]
        data = values[:,1:]
        #if value == "__EVAL_goal":
            #Multiplying goal value by 100
            #data = [ [t * 100 for t in listO] for listO in data]
        update = summarize_data(data,confidence)
        headerLine = []
        
        with open(evalFile, 'r') as f:
            first_line = f.readline()    

        headerLine.append("time")
        headerLine.append("mean")
        headerLine.append("ci_down")
        headerLine.append("ci_up")

        value = value.replace("EVAL","SUMMARY")
        with open(os.path.join(source, value), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerow((headerLine))
            csvfile.flush()

            for i in range(sum(trials.shape)):
                newrow = [trials[i]]
                for j in update.T[i]:
                    newrow.append("{:.2f}".format(j))
                csvwriter.writerow((newrow))
                csvfile.flush()


def cumulative_experiment_data(source,startingFrom=500,hfo=True):
    if hfo:
        values = ["__EVAL_steps", "__EVAL_rewards","__EVAL_goal"]
    else:
        values = ["__EVAL_steps", "__EVAL_rewards"]
    for value in values:
        evalFile = os.path.join(source, value)
        #print(evalFile)
        #evalFileContent = np.loadtxt(open(evalFile, "rb"), skiprows=1, delimiter=",", unpack=True)
        evalFileContent = pd.read_csv(open(evalFile, "rb"), skiprows=0, delimiter=",")
        values = evalFileContent.values
        #filters episodes for which less than 20% of the repetitions exist
        values = filter_small_samples(values, 0.2)
        import operator
        listToSort = np.asarray(values).tolist()
        listToSort.sort(key=operator.itemgetter(0))
        values = np.array(listToSort)
        trials = values[:,0]
        data = values[:,1:]

        firstIndex = 0
        for rep in range(1,data.shape[0]):
            if trials[rep] > startingFrom:
                for index in range(data.shape[1]):
                    data[rep][index] = data[rep-1][index] + data[rep][index]
            else:
                firstIndex += 1
        
        
        update = summarize_data(data)
        headerLine = []
        
        with open(evalFile, 'r') as f:
            first_line = f.readline()    
        if first_line.split(',')[0]=='steps':
            episodes = False
        else:
            episodes = True
        
        if episodes:
            headerLine.append("episodes")
        else:
            headerLine.append("step")
        headerLine.append("mean")
        headerLine.append("ci_down")
        headerLine.append("ci_up")

        value = value.replace("EVAL","CUMULATIVE")
        #if value == "__CUMULATIVE_goal":
            #Multiplying goal value by 100
        #    data = [ [t * 100 for t in listO] for listO in data]
        with open(os.path.join(source, value), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerow((headerLine))
            csvfile.flush()


            for i in range(firstIndex,sum(trials.shape)):
                newrow = [trials[i]]
                for j in update.T[i]:
                    newrow.append("{:.2f}".format(j))
                csvwriter.writerow((newrow))
                csvfile.flush()
        #Generates shifted graph (erasing steps in source tasks).
        #Finds initial step
        evalFileContent = pd.read_csv(open(evalFile, "rb"), skiprows=0, delimiter=",")
        values = evalFileContent.values
        values = filter_small_samples(values, 0.2)
        import operator
        listToSort = np.asarray(values).tolist()
        listToSort.sort(key=operator.itemgetter(0))
        values = np.array(listToSort)
        trials = values[:,0]
        data = values[:,1:]
        subtract = trials[0]
        #Shifts all the void steps
        for i in range(len(trials)):
                trials[i] = trials[i] - subtract
        for rep in range(1, data.shape[0]):
                for index in range(data.shape[1]):
                    data[rep][index] = data[rep - 1][index] + data[rep][index]
        update = summarize_data(data)
        #Genrates file
        value = value.replace("CUMULATIVE", "CUM_SHIFTED")
        with open(os.path.join(source, value), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerow((headerLine))
            csvfile.flush()

            for i in range(sum(trials.shape)):
                newrow = [trials[i]]
                for j in update.T[i]:
                    newrow.append("{:.2f}".format(j))
                csvwriter.writerow((newrow))
                csvfile.flush()
def draw_diff_graph(source1 = None, source2 = None,
               source3 = None, source4 = None,
               source5 = None, source6 = None, 
               path1 = None, name1 = "Algo1",
               path2 = None, name2 = "Algo2",
               path3 = None, name3 = "Algo3",
               path4 = None, name4 = "Algo4",
               path5 = None, name5 = "Algo5",
               path6 = None, name6 = "Algo5", ci = True,nCol = 1,
               #Parameters introduced to allow plot control
               xMin = None, xMax = None, yMin=None, yMax=None,bigFont=False,markEvery=None,
               ):
    plt.figure(figsize=(20,6), dpi=300)
    #Background
    plt.gca().set_facecolor('white')
    plt.grid(True,color='0.8')

    colors = ['#d7191c','#1a9641','#fdae61','#a6d96a','#fee08b','#d9ef8b']
    # 7570b3 #e7298a #66a61e #e6ab02
    
    lineWidth = 8.0 if bigFont else 4.0   

    with open(os.path.join(source1, path1), 'r') as f:
            first_line = f.readline()    
    #if first_line.split(',')[0]=='steps':
    #        episodes = False
    #else:
    #        episodes = True
    episodes = False

    markerSize = 20 if bigFont else 8

    
    if path1 != None:
        summary1File = os.path.join(source1, path1)
        summary1Content = np.loadtxt(open(summary1File, "rb"), skiprows=1, delimiter=",", unpack=True)
        X1 = summary1Content[0]
        Y11, Y12, Y13 = summary1Content[1],summary1Content[2],summary1Content[3]
        if ci:
            plt.fill_between(X1, Y11, Y12, facecolor=colors[0], alpha=0.2)
            plt.fill_between(X1, Y11, Y13, facecolor=colors[0], alpha=0.2)
        #if(not significant1 is None):
        #   plt.plot(X1,Y11,label=name1, color=colors[0], linewidth=lineWidth,markevery=significant1,marker="o",markersize=markerSize)
        #else:
        plt.plot(X1,Y11,label=name1, color=colors[0], linewidth=lineWidth,marker="o",markersize=markerSize,markevery=markEvery)

    if path2 != None:
        summary2File = os.path.join(source2, path2)
        summary2Content = np.loadtxt(open(summary2File, "rb"), skiprows=1, delimiter=",", unpack=True)
        X2 = summary2Content[0]
        Y21, Y22, Y23 = summary2Content[1],summary2Content[2],summary2Content[3]
        if ci:
            plt.fill_between(X2, Y21, Y22, facecolor=colors[1], alpha=0.2)
            plt.fill_between(X2, Y21, Y23, facecolor=colors[1], alpha=0.2)
        #if(not significant2 is None):
        #    plt.plot(X2,Y21,label=name2, color=colors[1], linewidth=lineWidth,markevery=significant2,marker="s",markersize=8)
        #else:
        plt.plot(X2,Y21,label=name2, color=colors[1], linewidth=lineWidth,marker="s",markersize=markerSize,markevery=markEvery)

    if path3 != None:
        summary3File = os.path.join(source3, path3)
        summary3Content = np.loadtxt(open(summary3File, "rb"), skiprows=1, delimiter=",", unpack=True)
        X3 = summary3Content[0]
        Y31, Y32, Y33 = summary3Content[1],summary3Content[2],summary3Content[3]
        if ci:
            plt.fill_between(X3, Y31, Y32, facecolor=colors[2], alpha=0.2)
            plt.fill_between(X3, Y31, Y33, facecolor=colors[2], alpha=0.2)
        #if(not significant3 is None):
        #    plt.plot(X3,Y31,label=name3, color=colors[2], linewidth=lineWidth,marker="D",markevery=significant3,markersize=8)
        #else:
        plt.plot(X3,Y31,label=name3, color=colors[2], linewidth=lineWidth,marker="D",markersize=markerSize,markevery=markEvery)

    if path4 != None:
        summary4File = os.path.join(source4, path4)
        summary4Content = np.loadtxt(open(summary4File, "rb"), skiprows=1, delimiter=",", unpack=True)
        X4 = summary4Content[0]
        Y41, Y42, Y43 = summary4Content[1],summary4Content[2],summary4Content[3]
        if ci:
            plt.fill_between(X4, Y41, Y42, facecolor=colors[3], alpha=0.2)
            plt.fill_between(X4, Y41, Y43, facecolor=colors[3], alpha=0.2)
        #if(not significant4 is None):
        #    plt.plot(X4,Y41,label=name4, color=colors[3], linewidth=lineWidth,markevery=significant4,marker="*",markersize=8)
        #else:
        plt.plot(X4,Y41,label=name4, color=colors[3], linewidth=lineWidth,marker="*",markersize=markerSize+4,markevery=markEvery)

    if path5 != None:
        summary5File = os.path.join(source5, path5)
        summary5Content = np.loadtxt(open(summary5File, "rb"), skiprows=1, delimiter=",", unpack=True)
        X5 = summary5Content[0]
        Y51, Y52, Y53 = summary5Content[1],summary5Content[2],summary5Content[3]
        if ci:
            plt.fill_between(X5, Y51, Y52, facecolor=colors[4], alpha=0.2)
            plt.fill_between(X5, Y51, Y53, facecolor=colors[4], alpha=0.2)
        #if(not significant5 is None):
        #    plt.plot(X5,Y51,label=name5, color=colors[4], linewidth=lineWidth,markevery=significant5,marker="x",markersize=8)
        #else:
        plt.plot(X5,Y51,label=name5, color=colors[4], linewidth=lineWidth,marker="x",markersize=markerSize,markevery=markEvery)
            
    if path6 != None:
        summary6File = os.path.join(source6, path6)
        summary6Content = np.loadtxt(open(summary6File, "rb"), skiprows=1, delimiter=",", unpack=True)
        X6 = summary6Content[0]
        Y61, Y62, Y63 = summary6Content[1],summary6Content[2],summary6Content[3]
        if ci:
            plt.fill_between(X6, Y61, Y62, facecolor=colors[5], alpha=0.2)
            plt.fill_between(X6, Y61, Y63, facecolor=colors[5], alpha=0.2)
        #if(not significant6 is None):
        ##    plt.plot(X6,Y61,label=name6, color=colors[5], linewidth=lineWidth,markevery=significant6,marker="^",markersize=8)
        #else:
        plt.plot(X6,Y61,label=name6, color=colors[5], linewidth=lineWidth,marker="^",markersize=markerSize)
            
    if not yMin is None:
            plt.ylim([yMin,yMax])
    if not xMin is None:
            plt.xlim([xMin,xMax])
            
    axisSize = 26 if bigFont else 18
    fontSize = 32 if bigFont else 20
    

    plt.ylabel('Difference in Rewards', fontsize=fontSize, fontweight='bold')
    plt.xlabel('Learning Steps', fontsize=fontSize, fontweight='bold')
    plt.legend(loc='best',prop={'size':fontSize, 'weight':'bold'},ncol=nCol)
    plt.tick_params(axis='both', which='major', labelsize=axisSize)
    plt.show()
    
                

def draw_graph(source1 = None, name1 = "Algo1",
               source2 = None, name2 = "Algo2",
               source3 = None, name3 = "Algo3",
               source4 = None, name4 = "Algo4",
               source5 = None, name5 = "Algo5",
               source6 = None, name6 = "Algo5",
               what = "__SUMMARY_reward", ci = True,nCol = 1,
               #Parameters introduced to allow plot control
               xMin = None, xMax = None, yMin=None, yMax=None,bigFont=False,markEvery=None,
               ):
    plt.figure(figsize=(20,6), dpi=300)
    #Background
    plt.gca().set_facecolor('white')
    plt.grid(True,color='0.8')

    colors = ['#d7191c','#1a9641','#fdae61','#a6d96a','#fee08b','#d9ef8b']
    # 7570b3 #e7298a #66a61e #e6ab02
    
    lineWidth = 8.0 if bigFont else 4.0   

    with open(os.path.join(source1, what), 'r') as f:
            first_line = f.readline()    
    #if first_line.split(',')[0]=='steps':
    #        episodes = False
    #else:
    #        episodes = True
    episodes = False

    markerSize = 20 if bigFont else 8

    
    if source1 != None:
        summary1File = os.path.join(source1, what)
        summary1Content = np.loadtxt(open(summary1File, "rb"), skiprows=1, delimiter=",", unpack=True)
        X1 = summary1Content[0]
        Y11, Y12, Y13 = summary1Content[1],summary1Content[2],summary1Content[3]
        if ci:
            plt.fill_between(X1, Y11, Y12, facecolor=colors[0], alpha=0.2)
            plt.fill_between(X1, Y11, Y13, facecolor=colors[0], alpha=0.2)
        #if(not significant1 is None):
        #   plt.plot(X1,Y11,label=name1, color=colors[0], linewidth=lineWidth,markevery=significant1,marker="o",markersize=markerSize)
        #else:
        plt.plot(X1,Y11,label=name1, color=colors[0], linewidth=lineWidth,marker="o",markersize=markerSize,markevery=markEvery)

    if source2 != None:
        summary2File = os.path.join(source2, what)
        summary2Content = np.loadtxt(open(summary2File, "rb"), skiprows=1, delimiter=",", unpack=True)
        X2 = summary2Content[0]
        Y21, Y22, Y23 = summary2Content[1],summary2Content[2],summary2Content[3]
        if ci:
            plt.fill_between(X2, Y21, Y22, facecolor=colors[1], alpha=0.2)
            plt.fill_between(X2, Y21, Y23, facecolor=colors[1], alpha=0.2)
        #if(not significant2 is None):
        #    plt.plot(X2,Y21,label=name2, color=colors[1], linewidth=lineWidth,markevery=significant2,marker="s",markersize=8)
        #else:
        plt.plot(X2,Y21,label=name2, color=colors[1], linewidth=lineWidth,marker="s",markersize=markerSize,markevery=markEvery)

    if source3 != None:
        summary3File = os.path.join(source3, what)
        summary3Content = np.loadtxt(open(summary3File, "rb"), skiprows=1, delimiter=",", unpack=True)
        X3 = summary3Content[0]
        Y31, Y32, Y33 = summary3Content[1],summary3Content[2],summary3Content[3]
        if ci:
            plt.fill_between(X3, Y31, Y32, facecolor=colors[2], alpha=0.2)
            plt.fill_between(X3, Y31, Y33, facecolor=colors[2], alpha=0.2)
        #if(not significant3 is None):
        #    plt.plot(X3,Y31,label=name3, color=colors[2], linewidth=lineWidth,marker="D",markevery=significant3,markersize=8)
        #else:
        plt.plot(X3,Y31,label=name3, color=colors[2], linewidth=lineWidth,marker="D",markersize=markerSize,markevery=markEvery)

    if source4 != None:
        summary4File = os.path.join(source4, what)
        summary4Content = np.loadtxt(open(summary4File, "rb"), skiprows=1, delimiter=",", unpack=True)
        X4 = summary4Content[0]
        Y41, Y42, Y43 = summary4Content[1],summary4Content[2],summary4Content[3]
        if ci:
            plt.fill_between(X4, Y41, Y42, facecolor=colors[3], alpha=0.2)
            plt.fill_between(X4, Y41, Y43, facecolor=colors[3], alpha=0.2)
        #if(not significant4 is None):
        #    plt.plot(X4,Y41,label=name4, color=colors[3], linewidth=lineWidth,markevery=significant4,marker="*",markersize=8)
        #else:
        plt.plot(X4,Y41,label=name4, color=colors[3], linewidth=lineWidth,marker="*",markersize=markerSize+4,markevery=markEvery)

    if source5 != None:
        summary5File = os.path.join(source5, what)
        summary5Content = np.loadtxt(open(summary5File, "rb"), skiprows=1, delimiter=",", unpack=True)
        X5 = summary5Content[0]
        Y51, Y52, Y53 = summary5Content[1],summary5Content[2],summary5Content[3]
        if ci:
            plt.fill_between(X5, Y51, Y52, facecolor=colors[4], alpha=0.2)
            plt.fill_between(X5, Y51, Y53, facecolor=colors[4], alpha=0.2)
        #if(not significant5 is None):
        #    plt.plot(X5,Y51,label=name5, color=colors[4], linewidth=lineWidth,markevery=significant5,marker="x",markersize=8)
        #else:
        plt.plot(X5,Y51,label=name5, color=colors[4], linewidth=lineWidth,marker="x",markersize=markerSize,markevery=markEvery)
            
    if source6 != None:
        summary6File = os.path.join(source6, what)
        summary6Content = np.loadtxt(open(summary6File, "rb"), skiprows=1, delimiter=",", unpack=True)
        X6 = summary6Content[0]
        Y61, Y62, Y63 = summary6Content[1],summary6Content[2],summary6Content[3]
        if ci:
            plt.fill_between(X6, Y61, Y62, facecolor=colors[5], alpha=0.2)
            plt.fill_between(X6, Y61, Y63, facecolor=colors[5], alpha=0.2)
        #if(not significant6 is None):
        ##    plt.plot(X6,Y61,label=name6, color=colors[5], linewidth=lineWidth,markevery=significant6,marker="^",markersize=8)
        #else:
        plt.plot(X6,Y61,label=name6, color=colors[5], linewidth=lineWidth,marker="^",markersize=markerSize)
            
    if not yMin is None:
            plt.ylim([yMin,yMax])
    if not xMin is None:
            plt.xlim([xMin,xMax])
            
    axisSize = 26 if bigFont else 18
    fontSize = 32 if bigFont else 20
    

    if what == "__SUMMARY_steps":
        #plt.title('Goal Percentage per Trial')
        plt.ylabel('Steps until completed', fontsize=fontSize, fontweight='bold')
    elif what == "__SUMMARY_reward" or what== "__SHIFTED_reward":
        #plt.title('Goal Percentage per Trial')
        plt.ylabel('Sum of Rewards', fontsize=fontSize, fontweight='bold')
    elif what == "__CUMULATIVE_reward":
        #plt.title('Goal Percentage per Trial')
        plt.ylabel('Cumul. Disc. Reward', fontsize=fontSize, fontweight='bold')
    elif what in ["__SUMMARY_goal","__SHIFTED_goal"]:
        plt.ylabel('Goal Percentage', fontsize=fontSize, fontweight='bold')
    elif what in ["__CUMULATIVE_goal","__CUM_SHIFTED_goal"]:
        plt.ylabel('Cumulative Goals', fontsize=fontSize, fontweight='bold')
    else:
        #plt.title('Unknown')
        plt.ylabel('Unknown')
    prefix = ""
    if "SHIFTED" in what:
        prefix = "Shifted "
    if episodes:
        plt.xlabel(prefix+'Learning Episodes', fontsize=fontSize, fontweight='bold')
    else:
        plt.xlabel(prefix+'Learning Steps', fontsize=fontSize, fontweight='bold')
    plt.legend(loc='best',prop={'size':fontSize, 'weight':'bold'},ncol=nCol)
    plt.tick_params(axis='both', which='major', labelsize=axisSize)
    plt.show()

def filter_small_samples(data,percentage):
    """
      ------------------------------------------------------------------------------------------------------------------
      Evaluates the data and removes episodes when they have less than <percentage>% of points when compared to the maximum found
      ------------------------------------------------------------------------------------------------------------------
    """
    # Number of repetitions in which this step was seen
    countValids = [np.count_nonzero(~np.isnan(data[x])) - 1 for x in range(len(data))]
    maxCount = max(countValids)
    # deletes positions for good if the number of examples is less than 20% of the repetitions
    data = [data[index] for index, value in enumerate(countValids) if value > percentage * maxCount]
    return data



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--source',default='/Users/leno/gitProjects/Curriculum_HFO/src/log/HFODomain/SARSA-NoneCurriculum')
    parser.add_argument('-r','--runs',type=int, default=50)
    return parser.parse_args()

def main():
    parameter = get_args()

    #--------
    fileFolder = './'

    source1 = fileFolder + 'qLearning'
    #source2 = fileFolder + 'PITAMSARSA-PrunedCurriculum'
    #source3 = fileFolder + 'VFReuseQLearning-ObjectOrientedCurriculum'
    #source4 = fileFolder + 'VFReuseQLearning-SvetlikCurriculum'
    #--------
    #startingCumulative = 5000
    #collect_experiment_data(source1, runs=2000, hfo=False)
    #summarize_experiment_data(source1, hfo=False)
    #cumulative_experiment_data(source1, startingFrom=startingCumulative, hfo=False)
    collect_experiment_data(source1, runs=50)
    summarize_experiment_data(source1)
    #cumulative_experiment_data(source2, startingFrom=startingCumulative, hfo=True)
    #collect_experiment_data(source3, runs=2000, hfo=False)
    #summarize_experiment_data(source3, hfo=False)
    #cumulative_experiment_data(source3, startingFrom=startingCumulative, hfo=False)
    #collect_experiment_data(source4, runs=2000)
    #summarize_experiment_data(source4)
    #cumulative_experiment_data(source4, startingFrom=startingCumulative)

    #draw_graph(source1=source1, source2=source2, source3=source3,
     #                    source4=source4,  # source5=source5,name5=name5,source6=source6,name6=name6,
     #                    ci=True, yMin=-15, yMax=20, xMin=0, xMax=4500)

    #collect_experiment_data(source=parameter.source, runs=parameter.runs)
    #summarize_experiment_data(parameter.source)
    #cumulative_experiment_data(parameter.source,startingFrom = 6000)

if __name__ == '__main__':
    main()
