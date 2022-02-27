#adding this command
from tkinter.ttk import Combobox

import numpy as np
import pandas as pd
import matplotlib
import io
import sys
from matplotlib.gridspec import GridSpec
from pandas._libs.parsers import is_float_dtype
from pandas.core.dtypes.common import is_int64_dtype, is_object_dtype
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def importFile(file_path, column_names = None):
    if column_names != '': column_names = stringToList(column_names)
    global df
    global features
    if (type(column_names) == list): #check to see if column names are provided
        df = pd.read_csv(file_path, names = column_names)
        features = column_names
    else:
        df = pd.read_csv(file_path)
        features = df.columns.values.tolist()

    dfLabel = Label(dataFrameFrame,text='Data set:')
    global dfContent
    dfContent = Text(dataFrameFrame)
    dfLabel.pack(side = LEFT)
    dfContent.pack(side = LEFT)
    dfContent.insert(INSERT,df)

    outputEntry = StringVar()
    labelImportWindow.configure(text = "Output Feature:")
    entryImportWindow.destroy()
    comboboxImportWindow = Combobox(master = import_window, value = features, textvariable=outputEntry)
    comboboxImportWindow.grid(row=0,column=1)
    btnImportWindow.config(text = 'Okay',command = lambda: setOutput(outputEntry.get()))

def setOutput(outputEntry):
    import_window.destroy()
    global output
    output = outputEntry

    # config button on the main frame after output is set
    importBtn.configure(state='disable')

    # show data info
    showDataInfo(df, display_info = True, display_unique_values='All')
    # activate the buttons
    visualizeBtn.config(state = 'active')
    preprocessBtn.config(state = 'active')
    modelBtn.config(state = 'active')

def showDataInfo(df, display_info = False, display_unique_values = None):
    #data_info_window.destroy()
    widgetAction(datainfoFrame,'clear')
    widgetAction(featureDetailFrame, 'clear')

    if (display_info == True):
        #datainfoFrame.pack()
        dataInfoLabel = Label(datainfoFrame,text='Data Frame Info:')
        displayInfoText = Text(datainfoFrame)
        dataInfoLabel.pack(side = LEFT)
        displayInfoText.pack(side = LEFT)
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        df.info(verbose=True, memory_usage='deep', show_counts=True, null_counts=None)
        dataInfo = new_stdout.getvalue()
        sys.stdout = old_stdout
        displayInfoText.insert(INSERT, dataInfo)
        displayInfoText.insert(INSERT, 'Output Column: ' + output)
    if display_unique_values != None:
        #featureDetailFrame.pack()
        uniqueValLabel = Label(featureDetailFrame,text='Feature Category:')
        uniqueValText = Text(featureDetailFrame)
        uniqueValLabel.pack(side = LEFT)
        uniqueValText.pack(side = LEFT)
        #display_unique_values = stringToList(display_unique_values)
        if display_unique_values == 'All':
            featuresToApply = df.columns
        else:
            featuresToApply = list(display_unique_values)
        for col in featuresToApply:
            if df[col].isnull().values.any() == True:
                uniqueValText.insert(INSERT, f'{col}: contains missing value, needs handling\n\n')
            elif is_int64_dtype(df[col]) == True or is_float_dtype(df[col]) == True:
                max = df[col].max()
                min = df[col].min()
                median = df[col].median()
                mean = df[col].mean()
                mode = df[col].mode()
                std = df[col].std()
                uniqueValText.insert(INSERT, f'{col}: Max:{max}, Min:{min}, Median:{median},\n'
                                             f' Mean:{mean:.2f}, std:{std:.2f} Mode:{mode}\n\n')
            elif is_object_dtype(df[col]) == True:
                uniqueVal, uniqueCount = np.unique(df.loc[:,col],return_counts=True)
                uniqueValText.insert(INSERT, f'{col}({len(uniqueVal)} unique values)\n')
                for value, count in zip(uniqueVal,uniqueCount):
                    uniqueValText.insert(INSERT,f'{value}: {count}')
                # uniqueValText.insert(INSERT, f'{col}: {uniqueVal}\n# of unqiue value: {uniqueVal.count()}\n\n')


def visualizeData(df, plot_column_size = 4, plot_type = None, feature_selected = None):
    featuresToApply = list(feature_selected)
    subPlotList = creatingSubplots(featuresToApply, plot_column_size)

    for feat,subPlot in zip(featuresToApply, subPlotList):
        if plot_type == 'scatter':  sns.scatterplot(data=df, x=feat, y=output, ax=subPlot, legend='auto')
        elif plot_type == 'histo':  sns.histplot(data=df, x=feat, hue=output, multiple='stack', ax=subPlot)
        elif plot_type == 'box':  sns.boxplot(x=df[feat],ax=subPlot)
        elif plot_type == 'pair':  sns.pairplot(df, hue=output, vars=featuresToApply)
    plt.show()
def preprocessing2(dataframe, handle_missing_values = None, one_hot_encode = None, ordinal_encode = None,
                  apply_log=None, remove_outlier = None,min_max_scaler = None, remove_feature = None,
                  convert_feature = None, convert_to = None):
    df = dataframe
    if (apply_log != None):
        featuresToApply = selectingFeatures(df,apply_log)
        for f in featuresToApply:
            df[f]=np.log10(df[f]+1)
        print (f'applied log transformation to: {featuresToApply}')
    if (remove_outlier != None):
        featuresToApply = selectingFeatures(df,remove_outlier)
        for f in featuresToApply:
            zScore = stats.zscore(df[f])
            absZScore = np.abs(zScore)
            filteredEntries = (absZScore < 3).all()
            df[f] = df[filteredEntries]
        print (f'removed outlier for features: {featuresToApply}')
    return df

def preprocessing(dataframe, preprocess_type, feature_selected, encoder_key = None, convert_to = None,missing_value_handle = None,
                  value_to_remove = None, replace_value = None):
    df = dataframe
    featureSelected = list(feature_selected)
    if preprocess_type == 'featToRemove':
        df = df.drop(columns = featureSelected, inplace = False)
    elif preprocess_type == 'oneHotEncode':
        for f in featureSelected:
            df = encode_onehot(df,f)
    elif preprocess_type == 'ordinalEncode':
        oEncoder = OrdinalEncoder(categories = encoder_key)
        df[featureSelected] = oEncoder.fit_transform(df[featureSelected])
    elif preprocess_type == 'minMaxScaler':
        featureSelected = list(feature_selected)
        df[featureSelected] = MinMaxScaler().fit_transform(df[featureSelected])
    elif preprocess_type == 'dataTypeConvert':
        if convert_to == 'integer':
            df[featureSelected] = df[featureSelected].astype(int)
        elif convert_to == 'float':
            df[featureSelected] = df[featureSelected].astype(float)
        elif convert_to == 'string':
            df[featureSelected] = df[featureSelected].astype(str)
    elif preprocess_type == 'handleMissingValue':
        if missing_value_handle == 'removeMissingValue':
            df.dropna(subset=featureSelected, inplace = True)
        elif missing_value_handle == 'removeSpecificValue':
            df[featureSelected] = df[featureSelected].replace(value_to_remove,np.NaN)
            df.dropna(subset=featureSelected, inplace=True)
        elif missing_value_handle == 'replaceValue':
            print('replace values:', replace_value[0], replace_value[1])
            df[featureSelected] = df[featureSelected].replace(replace_value[0], replace_value[1])
    return df


def trainTestSplit(df,output,test_size = None, train_size = None, random_state = None, shuffle = True, stratify = None):
    from sklearn.model_selection import train_test_split
    x = df.loc[:,df.columns != output].values
    y = df.loc[:,df.columns == output].values.ravel()
    if stratify == True:
        stratify = y
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = test_size, train_size=train_size,random_state=random_state, shuffle=shuffle, stratify=stratify)
    return xTrain, xTest, yTrain, yTest

def fitAndEvaulateModel(xTrain, xTest, yTrain, yTest, model, metric = None):
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    evalMetric = None
    if metric == 'accuracy':
        evalMetric = accuracy_score(yTest, yPred)
    return evalMetric

def crossValidate(df, output, model, n_splits = 5, shuffle = False, random_state = None, metric = None):
    x = df.loc[:, df.columns != output].values
    y = df.loc[:, df.columns == output].values.ravel()
    metricList = []
    kf = StratifiedKFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
    for trainIndex, testIndex in kf.split(x,y): #looping through the train and test set indices for the splits
        evalMetric = fitAndEvaulateModel(x[trainIndex], x[testIndex], y[trainIndex], y[testIndex], model, metric = metric)
        metricList += [evalMetric]
    print(f'{model} : {n_splits} fold cross validation result: {np.mean(metricList):.3f}+/-{np.std(metricList):.3f}')

def creatingSubplots(feature_input, plot_column_size = 2, sharex = False, sharey = False):
    featNum = len(feature_input)
    plot_row_size = int(math.ceil(featNum/plot_column_size))
    fig, ax = plt.subplots(plot_row_size, plot_column_size, sharex = sharex, sharey = sharey)  # subpolt with # of rows and columns defined above
    subplotList = []  # store subplot axis in a linear array
    for i in range(0, plot_row_size):
        for j in range(0, plot_column_size):
            if plot_column_size == 1:
                subplotList += [ax[i]]
            elif plot_row_size == 1:
                subplotList += [ax[j]]
            elif plot_row_size > 1:
                subplotList += [ax[i, j]]
    return subplotList
def selectingFeatures(df, feature_input):
    featuresToApply = None
    if feature_input == 'All':
        featuresToApply = list(df.columns)
        featuresToApply.remove(output)
    elif (type(feature_input) == list):
        featuresToApply = feature_input
    elif (type(feature_input) == str):
        featuresToApply = [feature_input]
    elif type(feature_input) == tuple:
        featuresToApply = list(feature_input)
    print (f'features selected: {featuresToApply}')
    return featuresToApply
def encode_onehot(df, f):
    df2 = pd.get_dummies(df[f], prefix='', prefix_sep='').groupby(level=0, axis=1).max().add_prefix(f+' - ')
    df3 = pd.concat([df, df2], axis=1)
    df3 = df3.drop([f], axis = 1)
    return df3

def gridSearch(df, output, model, param_grid, scoring = None, n_jobs = None, refit = True, cv = 5, verbose = 0,
        pre_dispatch = None, error_score = np.nan, return_train_score = False, show_graph = None, print_results = None):
    x = df.loc[:, df.columns != output].values
    y = df.loc[:, df.columns == output].values.ravel()
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, n_jobs = n_jobs,  refit = refit,
                      cv = cv, verbose = verbose, pre_dispatch = pre_dispatch, error_score = error_score, return_train_score = return_train_score)

    gs.fit(x,y)
    print(gs.scorer_)
    if print_results != None:
        dfResults = pd.DataFrame(gs.cv_results_)
        print (dfResults.to_string())
        plotGridSearch(dfResults, param_grid = param_grid)
    return gs.best_params_

def plotGridSearch(df_results, param_grid):
    paramList = list(param_grid.keys()) #put grid keys to a list
    gridKeyHeaders = ['param_' + param for param in paramList] #convert the parameter to match the column headers cv_results
    print (gridKeyHeaders)
    gridKeyHeaders.reverse()
    print (gridKeyHeaders)
    df_results.sort_values(by = gridKeyHeaders, inplace=True)
    print (df_results.to_string())

    scoresMean = df_results['mean_test_score']
    #plot Grid Search Scores
    if len(paramList) == 1:
        scoresMean = np.array(scoresMean)
        plt.plot(param_grid[paramList[0]], scoresMean)
        plt.title('Grid Search Scores')
        plt.xlabel(paramList[0])
        plt.ylabel('CV Average Score')
        plt.legend()
        plt.show()
    elif len(paramList) == 2:
        # get Test Scores Mean and std for each grid search
        print(f'{param_grid[paramList[0]]} length: {len(param_grid[paramList[0]])}')
        print(f'{param_grid[paramList[1]]} length: {len(param_grid[paramList[1]])}')
        scoresMean = np.array(scoresMean).reshape(len(param_grid[paramList[1]]), len(param_grid[paramList[0]]))
        print(scoresMean)
        for idx, val in enumerate(param_grid[paramList[1]]):
            # print ('idx:',idx, '  val:',val)
            # print ('scoresMean:',scoresMean[idx,:])
            plt.plot(param_grid[paramList[0]], scoresMean[idx,:], '-o', label = paramList[1] + ': ' + str(val))
            plt.title('Grid Search Scores')
            plt.xlabel(paramList[0])
            plt.ylabel('CV Average Score')
            plt.legend()
        plt.show()
    elif len(paramList) >= 3:
        dfList = []
        subplotList = creatingSubplots(param_grid[paramList[2]],sharex = True, sharey = True) # number of subplots = unique values of paramList[2]
        for param, subplot in zip(param_grid[paramList[2]],subplotList):
            df = df_results[df_results[gridKeyHeaders[0]] == param] # seperating the df_result based on the unique values of paramList[2], gridKeyHeaders is the reverse of paramList
            dfList += [df]
            scoresMean = df['mean_test_score']
            scoresMean = np.array(scoresMean).reshape(len(param_grid[paramList[1]]), len(param_grid[paramList[0]]))
            for idx, val in enumerate(param_grid[paramList[1]]):
                subplot.plot(param_grid[paramList[0]], scoresMean[idx, :], '-o', label=paramList[1] + ': ' + str(val))
                subplot.set_title(f'Grid Search Scores: {paramList[2]}={param}')
                subplot.set_xlabel(paramList[0])
                subplot.set_ylabel('CV Average Score')
                subplot.legend()
        plt.show()
def makeList(start, end, step = 1, log = None, num_of_sample = None, num_of_log_sample = 5):
    if log == None:
        if num_of_sample != None:
            step = int((end - start)/num_of_sample) #determine the step to get the number of sample in the list
            print ('step:', step)
        if type(start) == int and type(end) == int and type(step) == int:
            return list(range(start, end, step))
        else:
            return np.arange(start,end,step)
    elif type(log) == int or type(log) == float:
        return np.logspace(start = start, stop = end, num = num_of_log_sample,base = log)
def stringToList(string,seperator=','):
    if string == 'All':
        return 'All'
    else:
        list = [x.strip() for x in string.split(',')]
        return list
def openFileWindow():
    #clear existing widgets in the root window
    for frame in mainFrame.winfo_children():
        widgetAction(frame, action = 'clear')
    widgetAction(buttonFrame, action = 'disable')


    global filePath
    filePath = filedialog.askopenfilename(initialdir='/Users/jidengcheng/Desktop/Machine Learning Data', title='Import File')
    filePathLabel = Label(filePathFrame,text='File Path:')
    filePathContent = Label(filePathFrame,text=filePath)
    filePathLabel.pack(side = LEFT)
    filePathContent.pack(side = LEFT)
    importBtn.config(state = 'active')

def importWindow():
    global import_window
    import_window = Toplevel(root)
    import_window.title('import')
    import_window.geometry('500x500')

    header=StringVar()

    global labelImportWindow; global entryImportWindow
    labelImportWindow = Label(import_window, text='header:')
    labelImportWindow.grid(row=0,column=0)
    entryImportWindow = Entry(import_window, textvariable = header)
    entryImportWindow.grid(row=0,column=1)
    global btnImportWindow
    btnImportWindow = Button(import_window, text ='Next', command=lambda: importFile(filePath,
                                                                                     column_names=header.get()))
    btnImportWindow.grid(row=5,column=2)
    closeBtn = Button(import_window, text='Close', command=import_window.destroy)
    closeBtn.grid(row=5,column=3)


def paramTuningWindow():
    param_tuning_window = Toplevel(root)
    param_tuning_window.title('data modeling')
    param_tuning_window.geometry('500x500')

def modelWindow2():
    model_window = Toplevel(root)
    model_window.title('data modeling')
    model_window.geometry('500x500')

    modelList = ['KNN Classifier', 'SVM Classifier', 'Decision Tree', 'Random Forest', 'Multi-Layered Perceptron', 'Gaussian Naive Bayes']

    modelName = StringVar(); numOfFold = IntVar()
    modelLabel = Label(model_window, text = 'Select Model').pack()
    modelComboBox = Combobox(model_window, value = modelList, textvariable = modelName, state = 'readonly')

    modelComboBox.bind('<<ComboboxSelected>>',lambda event:setParameters(modelName.get()))
    parameterFrame = Frame(model_window)
    kFoldValLabel = Label(model_window,text = 'Number of folds for cross validation:')
    kFoldValComboBox = Combobox(model_window,value = [5,10,20], textvariable = numOfFold)
    modelButtonFrame = Frame(model_window)
    modelComboBox.pack();modelButtonFrame.pack(side = BOTTOM);parameterFrame.pack()

    paramSubFrame1 = Frame(parameterFrame)
    paramSubFrame2 = Frame(parameterFrame)
    paramSubFrame3 = Frame(parameterFrame)
    paramSubFrame4 = Frame(parameterFrame)
    paramSubFrame1.pack()
    paramSubFrame2.pack()
    paramSubFrame3.pack()
    paramSubFrame4.pack()

    paramLabel1 = Label(paramSubFrame1)
    paramLabel2 = Label(paramSubFrame2)
    paramLabel3 = Label(paramSubFrame3)
    paramLabel4 = Label(paramSubFrame4)
    paramComboBox1 = Combobox(paramSubFrame1)
    paramComboBox2 = Combobox(paramSubFrame2)
    paramComboBox3 = Combobox(paramSubFrame3)
    paramComboBox4 = Combobox(paramSubFrame4)

    paramInteger1 = IntVar();paramInteger2 = IntVar();paramInteger3 = IntVar();paramInteger4 = IntVar()
    paramFloat1 = DoubleVar();paramFloat2 = DoubleVar();paramFloat3 = DoubleVar();paramFloat4 = DoubleVar()
    paramString1 = StringVar();paramString2 = StringVar();paramString3 = StringVar();paramString4 = StringVar()
    paramBoolean1 = BooleanVar();paramBoolean2 = BooleanVar();paramBoolean3 = BooleanVar();paramBoolean4 = BooleanVar()

    executeBtn = Button(modelButtonFrame, text = 'Execute', command=lambda:crossValidate(df,output,model=fitModel(modelName = modelName.get()),
                                                                                         n_splits=numOfFold.get(),
                                                                                         metric='accuracy'
                                                                                         )).pack(side = LEFT)
    cancelBtn = Button(modelButtonFrame,text = 'Cancel', command = lambda:model_window.destroy()).pack(side = LEFT)

    def fitModel(modelName):
        if modelName == 'KNN Classifier':
            return KNeighborsClassifier(n_neighbors=paramInteger1.get(), p=paramInteger2.get(),
                                          weights=paramString1.get())

        elif modelName == 'SVM Classifier':
            if paramString2.get() != 'scale' and paramString2.get() != 'auto':
                gamma = float(paramString2.get())
            else:
                gamma = paramString2.get()
            return SVC(C=paramFloat1.get(), kernel=paramString1.get(), gamma=gamma)
        elif modelName == 'Decision Tree':
            if paramString1.get() != 'None':
                maxDepth = int(paramString1.get())
            elif paramString1.get() == 'None':
                maxDepth = None
            if paramString2.get() == "None":
                maxFeatures = None
            elif paramString2.get() == 'auto' or paramString2.get() == 'sqrt' or paramString2.get() == 'log2':
                maxFeatures = paramString2.get()
            else:
                maxFeatures = int(paramString2.get())
            return DecisionTreeClassifier(max_depth=maxDepth,min_samples_split=paramInteger1.get(),
                                          min_samples_leaf=paramInteger2.get(),max_features=maxFeatures)
        elif modelName == 'Random Forest':
            if paramString1.get() != 'None':
                maxDepth = int(paramString1.get())
            elif paramString1.get() == 'None':
                maxDepth = None
            return RandomForestClassifier(n_estimators=paramInteger1.get(),max_depth=maxDepth,min_samples_split=paramInteger2.get(),
                                          min_samples_leaf=paramInteger3.get())
        elif modelName == 'Multi-Layered Perceptron':
            if paramString1.get() == '100':
                hiddenLayerTuple = (100,)
            else:
                hiddenLayerTuple = tuple(stringToList(paramString1.get(), seperator=','))
            if paramString2.get() == 'auto':
                batchSize = paramString2.get()
            else:
                batchSize = int(paramString2.get())
            return MLPClassifier(hidden_layer_sizes=hiddenLayerTuple, alpha=paramFloat1.get(), batch_size=batchSize,
                                 max_iter=paramInteger1.get())


    def setParameters(model):
        if model == 'KNN Classifier':
            labelText1 = 'n_neightbors'
            labelText2 = 'power parameter:\n1-manhattan distance\n2-euclidean distance\n3-minkowski distance'
            labelText3 = 'eights'
            comboboxValue1 = makeList(5,200,step=10);comboboxTextVar1 = paramInteger1;comboboxDefult1 = 0;comboboxState1='normal'
            comboboxValue2 = [1,2,3];comboboxTextVar2 = paramInteger2;comboboxDefult2 = 1;comboboxState2='readonly'
            comboboxValue3 = ['uniform','distance'];comboboxTextVar3 = paramString1;comboboxDefult3 = 0;comboboxState3='readonly'

        elif model == 'SVM Classifier':
            labelText1 = 'C (Regularization Parameter)'
            labelText2 = 'Kernel'
            labelText3 = 'Gamma'
            comboboxValue1 = [0.5,1.0,1.5,2.0,5.0,1.0];
            comboboxTextVar1 = paramFloat1;
            comboboxDefult1 = 0;
            comboboxState1 = 'normal'
            comboboxValue2 = ['linear','poly','rbf','sigmoid','precomputed'];
            comboboxTextVar2 = paramString1;
            comboboxDefult2 = 2;
            comboboxState2 = 'readonly'
            comboboxValue3 = ['scale','auto','0.0001','0.001','0.01','0.1','1','10'];
            comboboxTextVar3 = paramString2;
            comboboxDefult3 = 1;
            comboboxState3 = 'normal'
        elif model == 'Decision Tree':
            labelText1 = 'Max Depth'
            labelText2 = 'Min Samples Split'
            labelText3 = 'Min Samples Leaf'
            labelText4 = 'Max Features'
            comboboxValue1 = ['None','10','15','20','30']
            comboboxTextVar1 = paramString1;
            comboboxDefult1 = 0;
            comboboxState1 = 'normal'
            comboboxValue2 = [2,5,10,15,20,30];
            comboboxTextVar2 = paramInteger1;
            comboboxDefult2 = 0;
            comboboxState2 = 'normal'
            comboboxValue3 = [1,2,5,10,15,20,30];
            comboboxTextVar3 = paramInteger2;
            comboboxDefult3 = 0;
            comboboxState3 = 'normal'
            comboboxValue4 = ['None', 'auto','sqrt','log2','2','5','8','10']
            comboboxTextVar4 = paramString2;
            comboboxDefult4 = 0;
            comboboxState4 = 'normal'
        elif model == 'Random Forest':
            labelText1 = 'Number of Trees'
            labelText2 = 'Max Depth'
            labelText3 = 'Min Samples Split'
            labelText4 = 'Min Samples Leaf'
            comboboxValue1 = [50,100,150,200]
            comboboxTextVar1 = paramInteger1;
            comboboxDefult1 = 1;
            comboboxState1 = 'normal'
            comboboxValue2 = ['None', '10', '15', '20', '30']
            comboboxTextVar2 = paramString1;
            comboboxDefult2 = 0;
            comboboxState2 = 'normal'
            comboboxValue3 = [2, 5, 10, 15, 20, 30];
            comboboxTextVar3 = paramInteger2;
            comboboxDefult3 = 0;
            comboboxState3 = 'normal'
            comboboxValue4 = [1, 2, 5, 10, 15, 20, 30];
            comboboxTextVar4 = paramInteger3;
            comboboxDefult4 = 0;
            comboboxState4 = 'normal'

        elif model == 'Multi-Layered Perceptron':
            labelText1 = 'Hidden Layer Sizes'
            labelText2 = 'Alpha (L2 penality)'
            labelText3 = 'Batch Size'
            labelText4 = 'Max Iteration'
            comboboxValue1 = ['100', '100,100']
            comboboxTextVar1 = paramString1;
            comboboxDefult1 = 0;
            comboboxState1 = 'normal'
            comboboxValue2 = [0.0001,0.001,0.01,0.1]
            comboboxTextVar2 = paramFloat1;
            comboboxDefult2 = 0;
            comboboxState2 = 'normal'
            comboboxValue3 = ['auto', '250','300'];
            comboboxTextVar3 = paramString2;
            comboboxDefult3 = 0;
            comboboxState3 = 'normal'
            comboboxValue4 = [50,150,200,250,300,500];
            comboboxTextVar4 = paramInteger1;
            comboboxDefult4 = 3;
            comboboxState4 = 'normal'
        paramLabel1.config(text=labelText1)
        paramLabel2.config(text=labelText2)
        paramLabel3.config(text=labelText3)
        paramComboBox1.config(value=comboboxValue1, textvariable=comboboxTextVar1, state=comboboxState1)
        paramComboBox1.current(comboboxDefult1)
        paramComboBox2.config(value=comboboxValue2, textvariable=comboboxTextVar2, state=comboboxState2)
        paramComboBox2.current(comboboxDefult2)
        paramComboBox3.config(value=comboboxValue3, textvariable=comboboxTextVar3, state=comboboxState3)
        paramComboBox3.current(comboboxDefult3)
        paramLabel1.pack(side=LEFT)
        paramComboBox1.pack(side=LEFT)
        paramLabel2.pack(side=LEFT)
        paramComboBox2.pack(side=LEFT)
        paramLabel3.pack(side=LEFT)
        paramComboBox3.pack(side=LEFT)
        if model == 'Decision Tree' or model == 'Random Forest' or model == 'Multi-Layered Perceptron':
            paramLabel4.config(text=labelText4)
            paramComboBox4.config(value=comboboxValue4, textvariable=comboboxTextVar4, state=comboboxState4)
            paramComboBox4.current(comboboxDefult4)
            paramLabel4.pack(side = LEFT)
            paramComboBox4.pack(side = LEFT)


        kFoldValLabel.pack();
        kFoldValComboBox.pack();
        kFoldValComboBox.current(0)


def modelWindow():
    model_window = Toplevel(root)
    model_window.title('data modeling')
    model_window.geometry('500x500')

    selectionFrame = Frame(model_window)
    parameterFrame = Frame(model_window)
    modelButtonFrame = Frame(model_window)
    selectionFrame.pack(); parameterFrame.pack(); modelButtonFrame.pack(side=BOTTOM)

    modelList = ['KNN Classifier', 'SVM Classifier', 'Decision Tree', 'Random Forest', 'Multi-Layered Perceptron']

    modelName = StringVar(); numOfFold = IntVar()
    modelLabel = Label(selectionFrame, text='Select Model')
    modelComboBox = Combobox(selectionFrame, value=modelList, textvariable=modelName, state='readonly')
    modelComboBox.bind('<<ComboboxSelected>>', lambda event: setParameters(modelName.get()))
    tuningBtn = Button(selectionFrame, text='Tune Parameters', command = lambda:parameterTuning(modelName.get()), state = 'disabled')
    modelLabel.pack(side = LEFT); modelComboBox.pack(side = LEFT); tuningBtn.pack(side = LEFT)
    kFoldValLabel = Label(model_window, text='Number of folds for cross validation:')
    kFoldValComboBox = Combobox(model_window, value=[5, 10, 20], textvariable=numOfFold)

    def parameterTuning(modelName):
        # new window for parameter tuning
        parameter_tuning_window = Toplevel(root)
        parameter_tuning_window.title('Parameter Tuning')
        parameter_tuning_window.geometry('500x500')

        # Setting up the static buttons and frame in the window
        tuningFrame = Frame(parameter_tuning_window)
        tuningButtonFrame = Frame(parameter_tuning_window)
        tuningFrame.pack(); tuningButtonFrame.pack(side = BOTTOM)

        tuningSubFrame1 = Frame(tuningFrame)
        tuningSubFrame2 = Frame(tuningFrame)
        tuningSubFrame3 = Frame(tuningFrame)
        tuningSubFrame4 = Frame(tuningFrame)
        tuningSubFrame1.pack();tuningSubFrame2.pack();tuningSubFrame3.pack();tuningSubFrame4.pack();

        tuneBtn = Button(tuningButtonFrame,text = 'Grid Search',command = lambda:entryStoring())
        cancelTuneBtn = Button(tuningButtonFrame,text = 'Cancel',command = parameter_tuning_window.destroy)
        tuneBtn.pack(side =LEFT); cancelTuneBtn.pack(side = LEFT)

        tuneLabelText1='dummy';tuneLabelText2='dummy';tuneLabelText3='dummy';tuneLabelText4 = 'dummy'
        entryStr1 = StringVar(); entryStr2 = StringVar(); entryStr3 = StringVar()
        intVal1 = IntVar(); intVal2 = IntVar(); intVal3 = IntVar()
        paramSelect1 = StringVar();paramSelect2 = StringVar();paramSelect3 = StringVar();paramSelect4 = StringVar();
        paramSelect5 = StringVar();paramSelect6 = StringVar();paramSelect7 = StringVar();paramSelect8 = StringVar();
        # the contents in the tuningFrame vary depending on the model selected
        if modelName == 'KNN Classifier':
            tuneLabelText1 = 'n_neightbors'
            tuneLabelText2 = 'power parameter:\n1-manhattan distance\n2-euclidean distance\n3-minkowski distance'
            tuneLabelText3 = 'weights'
            paramName1 = 'n_neighbors'
            paramName2 = 'p'
            paramName3 = 'weights'
            entryText1 = 'Linspace(start,end,step):'
            entryText2 = 'Multiple Selections:'
            entryText3 = 'Multiple Selections:'
            entry1 = Entry(tuningSubFrame1,textvariable = entryStr1)
            entry2 = Frame(tuningSubFrame2)
            entry2Select1 = Checkbutton(entry2,text='1-manhatten distance',variable=intVal1,onvalue=1,offvalue=-9999)
            entry2Select2 = Checkbutton(entry2, text='2-euclidean distance', variable=intVal2, onvalue=2,offvalue=-9999)
            entry2Select3 = Checkbutton(entry2, text='3-minkowski distance', variable=intVal3, onvalue=3,offvalue=-9999)
            entry2Select1.pack(side = LEFT);entry2Select2.pack(side = LEFT);entry2Select3.pack(side = LEFT);

            entry3 = Frame(tuningSubFrame3)
            entry3Select1 = Checkbutton(entry3, text='uniform', variable=paramSelect4, onvalue='uniform',offvalue='')
            entry3Select2 = Checkbutton(entry3, text='distance', variable=paramSelect5, onvalue='distance',offvalue='')
            entry3Select1.pack(side=LEFT);entry3Select2.pack(side=LEFT)
        elif modelName == 'SVM Classifier':
            tuneLabelText1 = 'C (Regularization Parameter)'
            tuneLabelText2 = 'Kernel'
            tuneLabelText3 = 'Gamma'
            paramName1 = 'C'
            paramName2 = 'kernel'
            paramName3 = 'gamma'
            entryText1 = 'Linspace(start,end,step):'
            entryText2 = 'Multiple Selections:'
            entryText3 = 'Linspace(start,end,step):'
            entry1 = Entry(tuningSubFrame1, textvariable=entryStr1)
            entry2 = Frame(tuningSubFrame2)
            entry2Select1 = Checkbutton(entry2, text='linear', variable=paramSelect1, onvalue='linear',
                                        offvalue='')
            entry2Select2 = Checkbutton(entry2, text='poly', variable=paramSelect2, onvalue='poly',
                                        offvalue='')
            entry2Select3 = Checkbutton(entry2, text='rbf', variable=paramSelect3, onvalue='rbf',
                                        offvalue='')
            entry2Select4 = Checkbutton(entry2, text='sigmoid', variable=paramSelect4, onvalue='sigmoid',
                                        offvalue='')
            entry2Select5 = Checkbutton(entry2, text='precomputed', variable=paramSelect5, onvalue='precomputed',
                                        offvalue='')
            entry2Select1.pack(side=LEFT);entry2Select2.pack(side=LEFT);entry2Select3.pack(side=LEFT);entry2Select4.pack(side=LEFT);entry2Select5.pack(side=LEFT)
            entry3 = Entry(tuningSubFrame3, textvariable=entryStr3)
        elif modelName == 'Decision Tree':
            tuneLabelText1 = 'Max Depth'
            tuneLabelText2 = 'Min Samples Split'
            tuneLabelText3 = 'Min Samples Leaf'
            tuneLabelText4 = 'Max Features'
            entryText1 = 'Linspace(start,end,step):'
            entryText2 = 'Linspace(start,end,step):'
            entryText3 = 'Linspace(start,end,step):'
            entryText4 = 'Linspace(start,end,step):'
        elif modelName == 'Random Forest':
            tuneLabelText1 = 'Number of Trees'
            tuneLabelText2 = 'Max Depth'
            tuneLabelText3 = 'Min Samples Split'
            tuneLabelText4 = 'Min Samples Leaf'
            entryText1 = 'Linspace(start,end,step):'
            entryText2 = 'Linspace(start,end,step):'
            entryText3 = 'Linspace(start,end,step):'
            entryText4 = 'Linspace(start,end,step):'
        elif modelName == 'Multi-Layered Perceptron':
            tuneLabelText1 = 'Hidden Layer Sizes'
            tuneLabelText2 = 'Alpha (L2 penality)'
            tuneLabelText3 = 'Batch Size'
            tuneLabelText4 = 'Max Iteration'
            entryText1 = 'Linspace(start,end,step):'
            entryText2 = 'Logspace(start,end,# of sample):'
            entryText3 = 'Linspace(start,end,step):'
            entryText4 = 'Linspace(start,end,step):'
        checkParam1 = StringVar();checkParam2 = StringVar();checkParam3 = StringVar();checkParam4 = StringVar();
        checkBtn1 = Checkbutton(tuningSubFrame1, text = tuneLabelText1, variable= checkParam1,onvalue =paramName1,offvalue ='')
        checkBtn2 = Checkbutton(tuningSubFrame2, text = tuneLabelText2, variable= checkParam2,onvalue =paramName2,offvalue ='')
        checkBtn3 = Checkbutton(tuningSubFrame3, text = tuneLabelText3, variable= checkParam3,onvalue =paramName3,offvalue ='')
        checkBtn1.pack(side = LEFT)
        checkBtn2.pack(side = LEFT)
        checkBtn3.pack(side = LEFT)
        entryLabel1 = Label(tuningSubFrame1, text = entryText1)
        entryLabel2 = Label(tuningSubFrame2, text = entryText2)
        entryLabel3 = Label(tuningSubFrame3, text = entryText3)
        entryLabel1.pack(side = LEFT)
        entryLabel2.pack(side = LEFT)
        entryLabel3.pack(side = LEFT)
        entry1.pack(side = LEFT);entry2.pack(side=LEFT); entry3.pack(side=LEFT)


        if modelName == 'Decision Tree' or modelName == 'Random Forest' or modelName == 'Multi-Layered Perceptron':
            checkBtn4 = Checkbutton(tuningSubFrame4, text = tuneLabelText4, variable= checkParam4,onvalue='' ,offvalue =None)
            checkBtn4.pack(side = LEFT)
            entryLabel4 = Label(tuningSubFrame4, text=entryText4)
            entryLabel4.pack(side=LEFT)


        def entryStoring():
            if modelName == 'KNN Classifier':
                model = KNeighborsClassifier()
                entryStr1List = list(map(int, stringToList(entryStr1.get(),
                                                           seperator=',')))  # turn the entry to list of strings, then convert all the strings to integers
                entry1List = list(range(entryStr1List[0], entryStr1List[1], entryStr1List[2]))
                entry2List = [intVal1.get(), intVal2.get(), intVal3.get()]
                entry3List = [paramSelect4.get(), paramSelect5.get()]
            elif modelName == 'SVM Classifier':
                model = SVC()
                entryStr1List = list(map(float, stringToList(entryStr1.get(),seperator=',')))
                entry1List = list(np.arange(entryStr1List[0], entryStr1List[1], entryStr1List[2]))
                entry2List = [paramSelect1.get(),paramSelect2.get(),paramSelect3.get(),paramSelect4.get(),paramSelect5.get()]
                entryStr3List = list(map(float, stringToList(entryStr3.get(), seperator=',')))
                entry3List = list(np.arange(entryStr3List[0], entryStr3List[1], entryStr3List[2]))
            paramNameList = [checkParam1.get(), checkParam2.get(), checkParam3.get()]
            paramDictList = [entry1List, entry2List, entry3List]
            gridSearchParam = dictionaryConvert(paramNameList, paramDictList)
            print(gridSearchParam)
            gridSearch(df,output,model = model, param_grid=gridSearchParam,show_graph=True,print_results=True)

    def dictionaryConvert(paramNameList, paramDictList):
        # processing the lists to drop 'None' values
        print ('new start------')
        print('len(paramNameList:', len(paramNameList))
        for i in range(len(paramNameList)-1, -1,-1):
            print('nameList i:',i, '  ', paramNameList[i])
            if paramNameList[i] == 0 or paramNameList[i] == '0' or paramNameList[i] == '':
                del paramNameList[i]
                del paramDictList[i]
        print('paraNameList', paramNameList)
        print('paraDictList', paramDictList)

        print('clean DictLists')
        for subDictList in paramDictList:
            print('subDictList:',subDictList,'  len(subDictList:', len(subDictList))
            try: # remove all unchecked string parameters
                while True:
                    subDictList.remove('')
            except ValueError:
                pass
            try: # remove all unchecked integer  parameters
                while True:
                    subDictList.remove(-9999)
            except ValueError:
                pass
        print ('paraNameList',paramNameList)
        print ('paraDictList',paramDictList)
        return dict(zip(paramNameList,paramDictList))


    paramSubFrame1 = Frame(parameterFrame)
    paramSubFrame2 = Frame(parameterFrame)
    paramSubFrame3 = Frame(parameterFrame)
    paramSubFrame4 = Frame(parameterFrame)
    paramSubFrame1.pack()
    paramSubFrame2.pack()
    paramSubFrame3.pack()
    paramSubFrame4.pack()

    paramLabel1 = Label(paramSubFrame1)
    paramLabel2 = Label(paramSubFrame2)
    paramLabel3 = Label(paramSubFrame3)
    paramLabel4 = Label(paramSubFrame4)
    paramComboBox1 = Combobox(paramSubFrame1)
    paramComboBox2 = Combobox(paramSubFrame2)
    paramComboBox3 = Combobox(paramSubFrame3)
    paramComboBox4 = Combobox(paramSubFrame4)

    paramInteger1 = IntVar();
    paramInteger2 = IntVar();
    paramInteger3 = IntVar();
    paramInteger4 = IntVar()
    paramFloat1 = DoubleVar();
    paramFloat2 = DoubleVar();
    paramFloat3 = DoubleVar();
    paramFloat4 = DoubleVar()
    paramString1 = StringVar();
    paramString2 = StringVar();
    paramString3 = StringVar();
    paramString4 = StringVar()
    paramBoolean1 = BooleanVar();
    paramBoolean2 = BooleanVar();
    paramBoolean3 = BooleanVar();
    paramBoolean4 = BooleanVar()

    executeBtn = Button(modelButtonFrame, text='Execute',
                        command=lambda: crossValidate(df, output, model=fitModel(modelName=modelName.get()),
                                                      n_splits=numOfFold.get(),
                                                      metric='accuracy'
                                                      )).pack(side=LEFT)
    cancelBtn = Button(modelButtonFrame, text='Cancel', command=lambda: model_window.destroy()).pack(side=LEFT)

    def fitModel(modelName):
        if modelName == 'KNN Classifier':
            return KNeighborsClassifier(n_neighbors=paramInteger1.get(), p=paramInteger2.get(),
                                        weights=paramString1.get())

        elif modelName == 'SVM Classifier':
            if paramString2.get() != 'scale' and paramString2.get() != 'auto':
                gamma = float(paramString2.get())
            else:
                gamma = paramString2.get()
            return SVC(C=paramFloat1.get(), kernel=paramString1.get(), gamma=gamma)
        elif modelName == 'Decision Tree':
            if paramString1.get() != 'None':
                maxDepth = int(paramString1.get())
            elif paramString1.get() == 'None':
                maxDepth = None
            if paramString2.get() == "None":
                maxFeatures = None
            elif paramString2.get() == 'auto' or paramString2.get() == 'sqrt' or paramString2.get() == 'log2':
                maxFeatures = paramString2.get()
            else:
                maxFeatures = int(paramString2.get())
            return DecisionTreeClassifier(max_depth=maxDepth, min_samples_split=paramInteger1.get(),
                                          min_samples_leaf=paramInteger2.get(), max_features=maxFeatures)
        elif modelName == 'Random Forest':
            if paramString1.get() != 'None':
                maxDepth = int(paramString1.get())
            elif paramString1.get() == 'None':
                maxDepth = None
            return RandomForestClassifier(n_estimators=paramInteger1.get(), max_depth=maxDepth,
                                          min_samples_split=paramInteger2.get(),
                                          min_samples_leaf=paramInteger3.get())
        elif modelName == 'Multi-Layered Perceptron':
            if paramString1.get() == '100':
                hiddenLayerTuple = (100,)
                print ('hidden layers:',hiddenLayerTuple)
            else:
                hiddenLayerTuple = tuple(map(int,stringToList(paramString1.get(), seperator=',')))
                print ('hidden layers:',hiddenLayerTuple)
            if paramString2.get() == 'auto':
                batchSize = paramString2.get()
            else:
                batchSize = int(paramString2.get())
            return MLPClassifier(hidden_layer_sizes=hiddenLayerTuple, alpha=paramFloat1.get(), batch_size=batchSize,
                                 max_iter=paramInteger1.get())

    def setParameters(model):
        tuningBtn.config(state = 'active') # turn on the tuning button
        if model == 'KNN Classifier':
            labelText1 = 'n_neightbors'
            labelText2 = 'power parameter:\n1-manhattan distance\n2-euclidean distance\n3-minkowski distance'
            labelText3 = 'weights'
            comboboxValue1 = makeList(5, 200, step=10);
            comboboxTextVar1 = paramInteger1;
            comboboxDefult1 = 0;
            comboboxState1 = 'normal'
            comboboxValue2 = [1, 2, 3];
            comboboxTextVar2 = paramInteger2;
            comboboxDefult2 = 1;
            comboboxState2 = 'readonly'
            comboboxValue3 = ['uniform', 'distance'];
            comboboxTextVar3 = paramString1;
            comboboxDefult3 = 0;
            comboboxState3 = 'readonly'

        elif model == 'SVM Classifier':
            labelText1 = 'C (Regularization Parameter)'
            labelText2 = 'Kernel'
            labelText3 = 'Gamma'
            comboboxValue1 = [0.5, 1.0, 1.5, 2.0, 5.0, 1.0];
            comboboxTextVar1 = paramFloat1;
            comboboxDefult1 = 0;
            comboboxState1 = 'normal'
            comboboxValue2 = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'];
            comboboxTextVar2 = paramString1;
            comboboxDefult2 = 2;
            comboboxState2 = 'readonly'
            comboboxValue3 = ['scale', 'auto', '0.0001', '0.001', '0.01', '0.1', '1', '10'];
            comboboxTextVar3 = paramString2;
            comboboxDefult3 = 1;
            comboboxState3 = 'normal'
        elif model == 'Decision Tree':
            labelText1 = 'Max Depth'
            labelText2 = 'Min Samples Split'
            labelText3 = 'Min Samples Leaf'
            labelText4 = 'Max Features'
            comboboxValue1 = ['None', '10', '15', '20', '30']
            comboboxTextVar1 = paramString1;
            comboboxDefult1 = 0;
            comboboxState1 = 'normal'
            comboboxValue2 = [2, 5, 10, 15, 20, 30];
            comboboxTextVar2 = paramInteger1;
            comboboxDefult2 = 0;
            comboboxState2 = 'normal'
            comboboxValue3 = [1, 2, 5, 10, 15, 20, 30];
            comboboxTextVar3 = paramInteger2;
            comboboxDefult3 = 0;
            comboboxState3 = 'normal'
            comboboxValue4 = ['None', 'auto', 'sqrt', 'log2', '2', '5', '8', '10']
            comboboxTextVar4 = paramString2;
            comboboxDefult4 = 0;
            comboboxState4 = 'normal'
        elif model == 'Random Forest':
            labelText1 = 'Number of Trees'
            labelText2 = 'Max Depth'
            labelText3 = 'Min Samples Split'
            labelText4 = 'Min Samples Leaf'
            comboboxValue1 = [50, 100, 150, 200]
            comboboxTextVar1 = paramInteger1;
            comboboxDefult1 = 1;
            comboboxState1 = 'normal'
            comboboxValue2 = ['None', '10', '15', '20', '30']
            comboboxTextVar2 = paramString1;
            comboboxDefult2 = 0;
            comboboxState2 = 'normal'
            comboboxValue3 = [2, 5, 10, 15, 20, 30];
            comboboxTextVar3 = paramInteger2;
            comboboxDefult3 = 0;
            comboboxState3 = 'normal'
            comboboxValue4 = [1, 2, 5, 10, 15, 20, 30];
            comboboxTextVar4 = paramInteger3;
            comboboxDefult4 = 0;
            comboboxState4 = 'normal'

        elif model == 'Multi-Layered Perceptron':
            labelText1 = 'Hidden Layer Sizes'
            labelText2 = 'Alpha (L2 penality)'
            labelText3 = 'Batch Size'
            labelText4 = 'Max Iteration'
            comboboxValue1 = ['100', '100,100']
            comboboxTextVar1 = paramString1;
            comboboxDefult1 = 0;
            comboboxState1 = 'normal'
            comboboxValue2 = [0.0001, 0.001, 0.01, 0.1]
            comboboxTextVar2 = paramFloat1;
            comboboxDefult2 = 0;
            comboboxState2 = 'normal'
            comboboxValue3 = ['auto', '250', '300'];
            comboboxTextVar3 = paramString2;
            comboboxDefult3 = 0;
            comboboxState3 = 'normal'
            comboboxValue4 = [50, 150, 200, 250, 300, 500];
            comboboxTextVar4 = paramInteger1;
            comboboxDefult4 = 3;
            comboboxState4 = 'normal'
        paramLabel1.config(text=labelText1)
        paramLabel2.config(text=labelText2)
        paramLabel3.config(text=labelText3)
        paramComboBox1.config(value=comboboxValue1, textvariable=comboboxTextVar1, state=comboboxState1)
        paramComboBox1.current(comboboxDefult1)
        paramComboBox2.config(value=comboboxValue2, textvariable=comboboxTextVar2, state=comboboxState2)
        paramComboBox2.current(comboboxDefult2)
        paramComboBox3.config(value=comboboxValue3, textvariable=comboboxTextVar3, state=comboboxState3)
        paramComboBox3.current(comboboxDefult3)
        paramLabel1.pack(side=LEFT)
        paramComboBox1.pack(side=LEFT)
        paramLabel2.pack(side=LEFT)
        paramComboBox2.pack(side=LEFT)
        paramLabel3.pack(side=LEFT)
        paramComboBox3.pack(side=LEFT)
        if model == 'Decision Tree' or model == 'Random Forest' or model == 'Multi-Layered Perceptron':
            paramLabel4.config(text=labelText4)
            paramComboBox4.config(value=comboboxValue4, textvariable=comboboxTextVar4, state=comboboxState4)
            paramComboBox4.current(comboboxDefult4)
            paramLabel4.pack(side=LEFT)
            paramComboBox4.pack(side=LEFT)

        kFoldValLabel.pack();
        kFoldValComboBox.pack();
        kFoldValComboBox.current(0)


def preprocessWindow():
    global preprocess_window
    preprocess_window = Toplevel(root)
    preprocess_window.title('Preprocessing')
    preprocess_window.geometry('500x500')

    radioButtonFrame = Frame(preprocess_window)
    listBoxFrame = Frame(preprocess_window)
    detailFrame = Frame(preprocess_window)
    buttonFrame = Frame(preprocess_window)
    radioButtonFrame.pack(side=LEFT)
    listBoxFrame.pack(side=LEFT)
    buttonFrame.pack(side=BOTTOM)


    # radio buttons
    preprocessAction = StringVar()
    featToRemoveBtn = Radiobutton(radioButtonFrame, text = 'remove feature', variable=preprocessAction, value='featToRemove',command = lambda:preprocessingDetail())
    oneHotEncodeBtn = Radiobutton(radioButtonFrame, text = 'one hot encode',variable=preprocessAction, value='oneHotEncode',command = lambda:preprocessingDetail())
    ordinalEncodeBtn = Radiobutton(radioButtonFrame, text = 'ordinal encode', variable=preprocessAction, value='ordinalEncode',command = lambda:preprocessingDetail(detail_needed=True))
    minMaxScalerBtn = Radiobutton(radioButtonFrame, text = 'normalize', variable=preprocessAction, value='minMaxScaler',command = lambda:preprocessingDetail())
    dataTypeConvertBtn = Radiobutton(radioButtonFrame, text = 'convert data type', variable=preprocessAction, value='dataTypeConvert', command=lambda:preprocessingDetail(detail_needed=True))
    handleMissingValueBtn = Radiobutton(radioButtonFrame, text = 'handle missing values', variable=preprocessAction, value='handleMissingValue', command=lambda:preprocessingDetail(detail_needed=True))
    featToRemoveBtn.pack()
    oneHotEncodeBtn.pack()
    ordinalEncodeBtn.pack()
    minMaxScalerBtn.pack()
    dataTypeConvertBtn.pack()
    handleMissingValueBtn.pack()

    #entry for the ordinal encoding details
    detailLabel = Label(detailFrame)
    detailLabel.pack()

    encoderKey = StringVar() #variable to store encoder key for ordinal encodig
    encoderEntry = Entry(detailFrame, textvariable = encoderKey)

    dataType = StringVar() #variable for the data type to convert to
    intBtn = Radiobutton(detailFrame, text='Integer', variable=dataType, value='integer')
    floatBtn = Radiobutton(detailFrame, text='Float', variable=dataType, value='float')
    stringBtn = Radiobutton(detailFrame, text='String', variable=dataType, value='string')

    missingValueHandle = StringVar(); valueToRemove = StringVar() #how to handle the missing value and the variable deemed as missing value
    removeNABtn = Radiobutton(detailFrame, text='Remove missing values', variable=missingValueHandle, value='removeMissingValue')
    removeValueFrame = Frame(detailFrame)
    removeNAShownAsBtn = Radiobutton(removeValueFrame, text = 'remove specific value:', variable=missingValueHandle,value='removeSpecificValue')
    valueToRemoveEntry = Entry(removeValueFrame, textvariable = valueToRemove)
    removeNAShownAsBtn.pack(side = LEFT); valueToRemoveEntry.pack(side = LEFT)

    valueToReplace = StringVar(); replaceToValue = StringVar()
    replaceValueFrame = Frame(detailFrame)
    replaceBtn = Radiobutton(replaceValueFrame, text = 'replace', variable = missingValueHandle, value='replaceValue')
    valueToReplaceEntry = Entry(replaceValueFrame, textvariable = valueToReplace)
    withLabel = Label(replaceValueFrame, text ='with')
    replaceToValueEntry = Entry(replaceValueFrame, textvariable = replaceToValue)
    replaceBtn.pack(side=LEFT);valueToReplaceEntry.pack(side=LEFT);withLabel.pack(side=LEFT);replaceToValueEntry.pack(side=LEFT)

    list = creatListBox(listBoxFrame) #list box for the preprocessing window

    executeBtn = Button(buttonFrame, text='Execute', command=lambda: selectAndShow(listbox=list, checked_item=preprocessAction.get(),
                                                                                   show_type='preprocessing', encoder_key=[stringToList(encoderKey.get(),',')],
                                                                                   data_type = dataType.get(),missing_value_handle=missingValueHandle.get(),
                                                                                   value_to_remove=valueToRemove.get(), replace_value = [valueToReplace.get(),replaceToValue.get()]))
    cancelBtn = Button(buttonFrame, text='Cancel', command=preprocess_window.destroy)
    executeBtn.pack(side = LEFT)
    cancelBtn.pack(side = RIGHT)

    def preprocessingDetail(detail_needed = False):
        detailFrame.pack_forget()
        intBtn.pack_forget(); floatBtn.pack_forget(); stringBtn.pack_forget()
        removeNABtn.pack_forget();removeValueFrame.pack_forget();valueToRemoveEntry.delete(0,END)
        replaceValueFrame.pack_forget(), valueToReplaceEntry.delete(0,END); replaceToValueEntry.delete(0,END)
        detailLabel.config(text = '')
        encoderEntry.pack_forget()
        if detail_needed == True:
            detailFrame.pack(side=LEFT)
            list.selection_clear(0,END)
            list.config(selectmode = SINGLE)
            list.bind("<<ListboxSelect>>",detailCallback)

        else:
            list.selection_clear(0, END)
            list.unbind("<<ListboxSelect>>")
            list.config(selectmode = MULTIPLE)

    def detailCallback(event):
        feature = list.get(list.curselection())
        print('featSelect:', feature)
        intBtn.pack_forget(); floatBtn.pack_forget(), stringBtn.pack_forget()
        removeNABtn.pack_forget(), removeValueFrame.pack_forget(),valueToRemoveEntry.delete(0,END)
        replaceValueFrame.pack_forget(), valueToReplaceEntry.delete(0,END); replaceToValueEntry.delete(0,END)
        if preprocessAction.get() == 'ordinalEncode':
            ordinalVar = np.unique(df[feature])
            ordinalVarStr = ''
            for i in ordinalVar:
                ordinalVarStr = ordinalVarStr + i + ','
            detailLabel.config(text='Put the ordinal variable in low to high order '
                                    '(separate by comma)\n for feature "' + feature + '"\n' + ordinalVarStr)
            encoderEntry.pack()
        elif preprocessAction.get() == 'dataTypeConvert':
            detailLabel.config(text='Convert data type of feature "' + feature + '" to:')
            intBtn.pack(); floatBtn.pack(); stringBtn.pack()
        elif preprocessAction.get() == 'handleMissingValue':
            numOfMissingValue = df[feature].isna().sum()
            print ('numOfMissingValue:',numOfMissingValue)
            detailLabel.config(text = 'Number of missing value (NaN) in "'+ feature + '":' + str(numOfMissingValue))
            removeNABtn.pack(), removeValueFrame.pack(), replaceValueFrame.pack()
def creatListBox(window):
    list = Listbox(window, selectmode = MULTIPLE)
    list.pack(expand = YES, fill = 'both')
    for i in features:
        if i == output: continue # skip the output feature, it shouldn't be selectable
        list.insert(END,i)
    return list

def selectAndShow(listbox, checked_item, show_type, encoder_key = None, data_type = None, missing_value_handle = None,
                  value_to_remove = None, replace_value = None):
    list = []
    for i in listbox.curselection():
        list.append(listbox.get(i))
    if show_type == 'data info': showDataInfo(df,display_info = checked_item, display_unique_values= list)
    elif show_type == 'visualization': visualizeData(df,plot_type=checked_item, feature_selected = list)
    elif show_type == 'preprocessing':
        preprocess_window.destroy()
        newdf = preprocessing(df,preprocess_type = checked_item, feature_selected = list, encoder_key = encoder_key,
                              convert_to=data_type, missing_value_handle = missing_value_handle, value_to_remove =value_to_remove,
                              replace_value = replace_value)
        refresh(newdf)
def visualWindow():
    global visual_window
    visual_window = Toplevel(root)
    visual_window.title("Visual Selection")
    visual_window.geometry('500x500')

    # frames for the visual_window
    visualFrame = Frame(visual_window)
    listBoxFrame = Frame(visual_window)
    buttonFrame = Frame(visual_window)
    visualFrame.pack(side = LEFT)
    listBoxFrame.pack(side = RIGHT)
    buttonFrame.pack(side = BOTTOM)

    # variables for the check buttons
    plotType = StringVar()
    histogramBtn = Radiobutton(visualFrame,text = 'histogram', variable = plotType, value = 'histo')
    scatterPlotBtn = Radiobutton(visualFrame,text = 'scatter plot', variable = plotType, value = 'scatter')
    boxPlotBtn = Radiobutton(visualFrame,text = 'box plot', variable = plotType, value = 'box')
    pairPlotBtn = Radiobutton(visualFrame,text = 'pair plot', variable = plotType, value = 'pair')
    histogramBtn.pack()
    scatterPlotBtn.pack()
    boxPlotBtn.pack()
    pairPlotBtn.pack()

    list = creatListBox(listBoxFrame)

    cancelBtn = Button(buttonFrame,text='Cancel',command=visual_window.destroy)
    okayBtn = Button(buttonFrame,text='Okay',command=lambda:selectAndShow(listbox = list, checked_item=plotType.get(), show_type='visualization'))
    okayBtn.pack(side = RIGHT)
    cancelBtn.pack(side = RIGHT)

def widgetAction(parent,action = None):
    for widget in parent.winfo_children():
        if action == 'clear':
            widget.destroy()
        elif action == 'disable':
            widget.config(state = 'disable')
        elif action == 'active':
            widget.config(state = 'active')
        elif action == 'hide':
            widget.pack_forget()
        elif action == 'show':
            widget.pack()

def refresh(dataframe):
    global df
    global features
    df = dataframe
    features = df.columns.values.tolist()
    dfContent.delete(1.0,END)
    dfContent.insert(INSERT,df)
    showDataInfo(df,display_info=True,display_unique_values=features)
from tkinter import *
from tkinter import filedialog

root = Tk('Dan\'s Machine Learning Pipeline')
root.geometry('{}x{}'.format(1000,1000))
root.title('Machine Learning Pipeline')
frameBorderWidth = 2
mainFrame = Frame(root,bg='blue',height = 300, width=300)
leftFrame = Frame(root, bg='red')
mainFrame.pack(side = LEFT)
leftFrame.pack(side = TOP)

filePathFrame = Frame(mainFrame, bg = 'green',borderwidth = frameBorderWidth)
dataFrameFrame = Frame(mainFrame, bg='yellow',borderwidth = frameBorderWidth)
datainfoFrame = Frame(mainFrame, bg = 'orange',borderwidth = frameBorderWidth)
featureDetailFrame = Frame(mainFrame, bg = 'red')
filePathFrame.pack()
dataFrameFrame.pack()
datainfoFrame.pack()
featureDetailFrame.pack()

openBtnFrame = Frame(leftFrame)
buttonFrame = Frame(leftFrame)
closeBtnFrame = Frame(leftFrame)
openBtnFrame.pack(side = TOP)
buttonFrame.pack()
closeBtnFrame.pack(side = BOTTOM)

openBtn = Button(openBtnFrame, text='Open File', command = openFileWindow)
openBtn.pack(side = TOP)
closeBtn = Button(closeBtnFrame, text='Close', command = root.destroy)
closeBtn.pack(side = BOTTOM)


# create the buttons for other function
importBtn = Button(buttonFrame, text='Import File', command=importWindow, state='disable')
visualizeBtn = Button(buttonFrame, text='Visualize', command=visualWindow, state = 'disable')
preprocessBtn = Button(buttonFrame, text='Preprocess', command=preprocessWindow, state = 'disable')
modelBtn = Button(buttonFrame, text='Model', command = modelWindow, state='disable')
paramTuningBtn = Button(buttonFrame, text ='Parameter Tuning', command = paramTuningWindow, state='disable')
# position the buttons
importBtn.pack();visualizeBtn.pack();preprocessBtn.pack();modelBtn.pack()

#root.mainloop()

exit()

#Classification dataSet
path_car = '/Users/jidengcheng/Desktop/Machine Learning Data/car.csv'
path_breast = '/Users/jidengcheng/Desktop/Machine Learning Data/breast-cancer-wisconsin.csv'
path_house = '/Users/jidengcheng/Desktop/Machine Learning Data/house-votes-84.csv'

header_car = ['buying','maint','doors','persons','lug_boot','safety','class']
header_breast = ['sample code','clump thickness','uniformity of cell size', 'uniformity of cell shape', 'marginal adhesion', 'single epithelial cell size','bare nuclei', 'bland chromatin', 'normal nucleoli', 'mitoses','class']
header_house = ['class', 'handicapped infants', 'water project cost sharing', 'adoption of the budget resolution', 'physician fee freeze', 'el salvador aid', 'religious groups in schools', 'anti satellite test ban', 'aid to nicaraguan contras', 'mx missile', 'immigration', 'synfuels corporation cutback', 'superfund right to sue', 'crime', 'duty free exports', 'export administration act south africa']
output_car = 'class'
output_breast = 'class'
output_house = 'class'

path_forest_fire ='/Users/jidengcheng/Desktop/Machine Learning Data/forestfires.csv'
header_forest_fire = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']
output_forest_fire = 'area'
ordinalEncoderKeys = [['buying','maint','doors','persons','lug_boot','safety'],
                [['vhigh','high','med','low'],['vhigh','high','med','low'],['2','3','4','5more'],
                    ['2','4','more'],['small','med','big'],['low','med','high']]]

# parameter variable assignment
path = path_car
output = output_car
header = header_car
df = importFile(path, column_names = header)
#showDataFrame(df,output_feature=output, display_dataframe=True, display_info=True, display_unique_values="All")
#visualizeData(df, histogram='All', scatterplot=None, plot_column_size = 3, pairplot=None)
df = preprocessing(df,handle_missing_values = None,one_hot_encode= None, ordinal_encode=ordinalEncoderKeys, apply_log=None,
                   remove_outlier=None, min_max_scaler=None, remove_feature=None, convert_feature=None,
                   convert_to=None)
#visualizeData(df, histogram=None, scatterplot=['temp','wind'], boxplot =None,pairplot = None)
#visualizeData(df, histogram=None, scatterplot='All', boxplot =None,pairplot = None)

#showDataFrame(df,output_feature=output, display_dataframe=True, display_info=True, display_unique_values="All")

xTrain, xTest, yTrain, yTest = trainTestSplit(df, output = output, test_size= 0.2, stratify = True)

x = df.loc[:, df.columns != output].values
y = df.loc[:, df.columns == output].values.ravel()

# the training set will run through different classifer to compare results
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
classifierInputList = []
#classifierInputList.append(SVC(kernel = 'linear', probability = True))
#classifierInputList.append(RandomForestClassifier())
#classifierInputList.append(MLPClassifier())
#classifierInputList.append(GaussianNB())
#classifierInputList.append(KNeighborsClassifier())

# for classifierInput in classifierInputList:
#     crossValidate(df,output,classifierInput, metric='accuracy')

parameters = {'n_neighbors':makeList(5,100,10),'weights':['uniform','distance']}
parameters = {'gamma':makeList(0.01,0.3,0.01),'C':makeList(-1,3,log=10, num_of_sample=5),'kernel':('rbf','poly')}
# parameters = {'C':makeList(-1,3,log=10, num_of_sample=5),'gamma':makeList(0.1,0.4,0.1)}
#parameters = {'gamma':makeList(0.1,0.4,0.1),'C':makeList(0,3,log=10, num_of_sample=4)}
print(gridSearch(df,output,SVC(),parameters, print_results= True))


