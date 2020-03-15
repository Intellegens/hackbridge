import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import pandas as pd

from DatasetAnalyzer import Dataset, BasicAnalyzer
import Alchemite

from importlib import reload
from copy import deepcopy
DEBUG = True

version = "alchemite"
height=800
width=1500

font_header = ('times', 20, 'bold')

root = tk.Tk()
root.title("Comparing {} to other ML algorithms".format(version))
root.geometry("{}x{}+0+0".format(width, height))

test = None
analyzer_basic = BasicAnalyzer()
analyzer_alchemite = Alchemite.AlchemiteAnalyzer(
    credentials=filedialog.askopenfilename(initialdir=".", title="Credentials"))

chosenMenu = None

plt.plot()

# Program Structure
# - Functionality
#   - load{}... on load of frame
#   - choose{}... on press of button
# - Widget definition
# - Widget linking



# FUNCTIONALITY

# FUNCTIONALITY - "Menu" Frame

def chooseMenu(choice):
    global chosenMenu, frames, analyzer_basic, height, width
    if chosenMenu != None:
        # frames[chosenMenu].pack_forget()
        frames[chosenMenu].grid_forget()
    chosenMenu = choice
    # frames[chosenMenu].pack(fill=tk.BOTH, expand=True)
    if chosenMenu == 'data':
        frames[chosenMenu].grid(row=0, column=0, sticky="n")
    else:
        frames[chosenMenu].grid(row=0, column=0, sticky="nsew")

    frames['main'].grid_columnconfigure(0, minsize=10, weight=1)
    frames['main'].grid_rowconfigure(0, minsize=10, weight=1)

    if choice == "basic":
        loadBasic()
    elif choice == "comparison":
        loadComparison()
    elif choice == "execute":
        loadExecute()
    elif choice == "outliers":
        loadOutliers()


# FUNCTIONALITY - "Data" Frame

def chooseData(value):
    global filenames, analyzer_basic
    # filenames.update({value:
    #    filedialog.askopenfilename(initialdir=".", title="Select {}".format(value))})
    filename = filedialog.askopenfilename(initialdir=".", title="Select {}".format(value))
    if filename is not None:
        data['values'][value].set(filename)
        if value == 'train':
            train_df = pd.read_csv(filename, index_col=0).astype('float32')
            print(train_df.head())
            train = Dataset(data['values']['name'].get(), 'train', train_df)
            BasicAnalyzer.setTrainDS(train)
        if value == 'valid':
            valid_df = pd.read_csv(filename, index_col=0).astype('float32')
            print(valid_df.head())
            valid = Dataset(data['values']['name'].get(), 'valid', valid_df)
            BasicAnalyzer.setValidDS(valid)
        if value == 'test':
            test_df = pd.read_csv(filename, index_col=0).astype('float32')
            print(test_df.head())
            test = Dataset(data['values']['name'].get(), 'test', test_df)
            BasicAnalyzer.setTestDS(test)


# FUNCTIONALITY - "Explore" Frame

state_explore = ''
def chooseExplore(choice):
    print("comparison.chooseExplore")
    global analyzer_basic, state_explore, canvas_widget
    if choice == "densities":
        if state_explore != 'density':
            explore['widgets']['info'].grid_forget()
            explore['widgets']['feature'].grid(row=0, column=0, sticky='nsew')
            explore['widgets']['feature']['menu'].delete(0, 'end')
            for feature in (BasicAnalyzer.train.getFeatures() if BasicAnalyzer.train is not None else []):
                explore['widgets']['feature']['menu'].add_command(label=feature, command=tk._setit(explore['values']['feature'], feature))
        chooseExplore2()
        state_explore = 'density'
    else:
        if state_explore != 'info':     # show the info text if density is currently displayed
            explore['widgets']['feature'].grid_forget()
            if canvas_widget:
                canvas_widget.destroy()
            explore['widgets']['info'].grid(row=0, column=0, sticky='nsew')
            explore['widgets']['main'].grid_rowconfigure(0, minsize=200, weight=1)
            explore['widgets']['main'].grid_columnconfigure(0, minsize=200, weight=1)
            state_explore = 'info'
        if choice == "dimensions":
            explore['widgets']['info'].delete(1.0, tk.END)
            explore['values']['info'] = analyzer_basic.printDimensions()
            explore['widgets']['info'].insert(tk.END, explore['values']['info'])
        if choice == "missing_vals":
            explore['widgets']['info'].delete(1.0, tk.END)
            explore['values']['info'] = analyzer_basic.printMissingValues()
            explore['widgets']['info'].insert(tk.END, explore['values']['info'])
        if choice == "basic_stats":
            explore['widgets']['info'].delete(1.0, tk.END)
            explore['values']['info'] = analyzer_basic.printBasicStats()
            explore['widgets']['info'].insert(tk.END, explore['values']['info'])

canvas_widget = None
def chooseExplore2(*args):
    global analyzer_basic, test, canvas_widget
    text = ""
    feature = explore['values']['feature'].get()

    if feature:
        # fig = analyzer_basic.plotDensities(test, features=[feature])
        datasets = []
        datasets += [BasicAnalyzer.train] if BasicAnalyzer.train is not None else []
        datasets += [BasicAnalyzer.valid] if BasicAnalyzer.valid is not None else []
        datasets += [test] if test is not None else []

        fig = Dataset.plotDensities(datasets, features=[feature])
        if canvas_widget:
            canvas_widget.destroy()
        canvas = FigureCanvasTkAgg(fig, explore['widgets']['main'])
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=1, rowspan=2, sticky='nsew')

        explore['widgets']['main'].grid_rowconfigure(0, minsize=50)
        explore['widgets']['main'].grid_rowconfigure(1, minsize=200, weight=1)
        explore['widgets']['main'].grid_columnconfigure(0, minsize=100)
        explore['widgets']['main'].grid_columnconfigure(1, minsize=200, weight=1)



# FUNCTIONALITY - "Basic" Frame

def loadBasic():
    print("comparison.loadBasic")
    global analyzer_alchemite

    text = ""
    if len(analyzer_basic.gcvs["models"]) != 0 and basic['values']['feature'].get() != "":
        text += analyzer_basic.getFeatureModelDescription(basic['values']['feature'].get())
    basic['widgets']['info'].delete(1.0, tk.END)
    basic['values']['info'] = text
    basic['widgets']['info'].insert(tk.END, basic['values']['info'])

    basic['widgets']['feature']['menu'].delete(0, 'end')
    for feature in (BasicAnalyzer.train.getFeatures() if BasicAnalyzer.train is not None else []):
        basic['widgets']['feature']['menu'].add_command(label=feature, command=tk._setit(basic['values']['feature'], feature))


def chooseBasic(choice):
    global analyzer_basic, analyzer_alchemite
    text = ""
    if choice == 'train':
        BasicAnalyzer.train.fillMissingValues(np.average)
        analyzer_basic.fitFeatureModels()
        BasicAnalyzer.train.restoreMissingValues()
    elif choice == 'train_alchemite':
        analyzer_alchemite.fitFeatureModels()
    elif choice == 'load':
        filename = filedialog.askopenfilename(initialdir=".", title="Load {}".format("Model"))
        analyzer_basic.loadGCV(filename)
    elif choice == 'save':
        filename = filedialog.asksaveasfilename(initialdir=".", title="Save {}".format("Model"))
        analyzer_basic.saveGCV(filename)
    elif choice == 'set_params':
        feature = basic['values']['feature'].get()
        if feature:
            param_code = basic['widgets']['info'].get(1.0,'end')
            analyzer_basic.params_code.update({feature: param_code})

    basic['widgets']['feature']['menu'].delete(0, 'end')
    for feature in (BasicAnalyzer.train.getFeatures() if BasicAnalyzer.train is not None else []):
        basic['widgets']['feature']['menu'].add_command(label=feature, command=tk._setit(basic['values']['feature'], feature))
    chooseBasic2()


def chooseBasic2(*args):
    global analyzer_basic
    text = ""
    feature = basic['values']['feature'].get()
    show = basic['values']['show'].get()

    if show != "Model":
        text += analyzer_basic.gcvs["params"][feature] if feature in analyzer_basic.gcvs["params"] else ""
    else:
        text += analyzer_basic.getFeatureModelDescription(feature) if feature in analyzer_basic.gcvs["models"] else ""

    basic['widgets']['info'].delete(1.0, tk.END)
    basic['values']['info'] = text
    basic['widgets']['info'].insert(tk.END, basic['values']['info'])

# FUNCTIONALITY - "Compare" Frame

def loadComparison():
    global analyzer_basic, analyzer_alchemite, test
    pass


def chooseComparison(choice):
    global analyzer_basic, analyzer_alchemite, test

    if choice=='compare':
        test.getImputationTestset(comparison['values']['errors'].get())
        test_basic_imp = Dataset(test.name, "basic_imp", test.imp_test_set.copy(deep=True))
        test_alchemite_imp = Dataset(test.name, "alchemite_imp", test.imp_test_set.copy(deep=True))

        analyzer_basic.iterateMissingValuePredictions(test_basic_imp, iterations=comparison['values']['iterations'].get())
        text_basic = "Basic model after {} iterations\n".format(BasicAnalyzer.iterations)
        text_basic += analyzer_basic.printImputedVsActual(test, test_basic_imp)

        analyzer_alchemite.iterateMissingValuePredictions(test_alchemite_imp)
        text_alchemite = "Alchemite\n"
        text_alchemite += analyzer_alchemite.printImputedVsActual(test, test_alchemite_imp)

        text_basic += "\n\nSetting Column to None"
        text_alchemite += "\n\nSetting Column to None"
        features = test.getFeatures()
        for feature in features:
            test.getImputationTestset(1, [feature])
            test_basic_imp = Dataset(test.name, "basic_imp_{}".format(feature),
                test.imp_test_set.copy(deep=True))
            test_alchemite_imp = Dataset(test.name, "alchemite_imp_{}".format(feature),
                test.imp_test_set.copy(deep=True))

            analyzer_basic.iterateMissingValuePredictions(
                test_basic_imp,
                iterations=comparison['values']['iterations'].get(),
                features=[feature])
            text_basic += analyzer_basic.printImputedVsActual(test, test_basic_imp, [feature], False)

            analyzer_alchemite.iterateMissingValuePredictions(test_alchemite_imp)
            text_alchemite += analyzer_alchemite.printImputedVsActual(test, test_alchemite_imp, [feature], False)

        comparison['widgets']['info_basic'].delete(1.0, tk.END)
        comparison['values']['info_basic'] = text_basic
        comparison['widgets']['info_basic'].insert(tk.END, comparison['values']['info_basic'])

        comparison['widgets']['info_alchemite'].delete(1.0, tk.END)
        comparison['values']['info_alchemite'] = text_alchemite
        comparison['widgets']['info_alchemite'].insert(tk.END, comparison['values']['info_alchemite'])
    elif choice =='export':
        if analyzer_basic.imputed is not None:
            filename = filedialog.asksaveasfilename(initialdir=".", title="Export Basic")
            analyzer_basic.saveImputedAlchemiteFormat(filename)
        if analyzer_alchemite.imputed is not None:
            filename = filedialog.asksaveasfilename(initialdir=".", title="Export Alchemite")
            analyzer_alchemite.saveImputedAlchemiteFormat(filename)

# FUNCTIONALITY - "Outlier" Frame

def loadOutliers():
    print("comparison.loadOutliers")
    global analyzer_basic, test

    outliers['widgets']['distribution_feature']['menu'].delete(0, 'end')
    for feature in (BasicAnalyzer.train.getFeatures() if BasicAnalyzer.train is not None else []):
        outliers['widgets']['distribution_feature']['menu'].add_command(label=feature, command=tk._setit(outliers['values']['distribution_feature'], feature))


prev_choice = None
def chooseOutliers(choice):
    print("comparison.chooseOutliers")
    global prev_choice

    if prev_choice is not None:
        outliers['widgets'][prev_choice].grid_forget()
    outliers['widgets'][choice].grid(row=0, column=0, sticky='nsew')
    outliers['widgets']['main'].grid_rowconfigure(0, minsize=50, weight=1)
    outliers['widgets']['main'].grid_columnconfigure(0, minsize=50, weight=1)
    prev_choice = choice

    if choice == 'distribution':
        print(1)
    elif choice == 'yshuffle':
        features = BasicAnalyzer.train.getFeatures()

        # clear the y-shuffle menu frame
        outliers['widgets']['yshuffle_run_basic'].grid_forget()
        outliers['widgets']['yshuffle_run_alchemite'].grid_forget()
        for feature, value in outliers['widgets']['yshuffle_features'].items():
            outliers['widgets']['yshuffle_features'][feature].grid_forget()
        outliers['values']['yshuffle_features'].clear()
        outliers['widgets']['yshuffle_features'].clear()

        outliers['widgets']['yshuffle_shuffle'].grid(row=0, column=0, sticky="nsew")
        outliers['widgets']['yshuffle_run_basic'].grid(row=1, column=0, sticky="nsew")
        outliers['widgets']['yshuffle_run_alchemite'].grid(row=2, column=0, sticky="nsew")
        for i, feature in enumerate(features):
            outliers['values']['yshuffle_features'].update({feature : tk.IntVar()})
            outliers['widgets']['yshuffle_features'].update(
                {feature : tk.Checkbutton(outliers['widgets']['yshuffle_menu'], text=feature, variable=outliers['values']['yshuffle_features'][feature])})
            outliers['widgets']['yshuffle_features'][feature].deselect()
            outliers['widgets']['yshuffle_features'][feature].grid(row=i+3, column=0, sticky='nsw')

        for j in range(i+4):
            outliers['widgets']['yshuffle_menu'].grid_rowconfigure(j, minsize=25, weight=1)
        outliers['widgets']['yshuffle_menu'].grid_columnconfigure(0, minsize=50, weight=1)
    elif choice == 'add3sigma':
        print(3)


def chooseOutliers2(*args):
    print("comparison.chooseOutliers2")

    global analyzer_basic, test, canvas_widget
    feature = outliers['values']['distribution_feature'].get()

    if feature:
        if not hasattr(test, "out"):
            _, _, test.out = analyzer_basic.findOutliers(test)

        fig = Dataset.plotDensities([test.out], features=[feature], x_range=[0.0, 1.0])
        if canvas_widget:
            canvas_widget.destroy()
        canvas = FigureCanvasTkAgg(fig, outliers['widgets']['distribution'])
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=1, column=0, rowspan=2, sticky='nsew')

        outliers['widgets']['distribution'].grid_rowconfigure(0, minsize=50)
        outliers['widgets']['distribution'].grid_rowconfigure(1, minsize=50, weight=1)
        outliers['widgets']['distribution'].grid_columnconfigure(0, minsize=200, weight=1)


yshuffled_idx, yshuffled = None, None
def outliers_yshuffle_run(choice):
    print("comparison.outliers_yshuffle_run")
    global analyzer_basic, analyzer_alchemite, test, yshuffled_idx, yshuffled
    text = ""
    yfeatures = []
    for feature, value in outliers['values']['yshuffle_features'].items():
        if value.get() == 1:
            yfeatures.append(feature)

    if choice == 'shuffle' or yshuffled_idx is None or yshuffled is None:
        yshuffled_idx, yshuffled = test.getYShuffledSet(yfeatures, 10)
        yshuffled_idx = [x for x, _ in yshuffled_idx]


    features_display = deepcopy(yfeatures)
    features_display += BasicAnalyzer.train.getFeaturesWithout(yfeatures)
    if choice == 'alchemite':
        report = analyzer_alchemite.findOutliers(yshuffled)
    elif choice == 'basic':
        estimated, estimated_std, estimated_pc, report = analyzer_basic.findOutliers(yshuffled)
    if choice == 'alchemite' or choice == 'basic':
        text += "1... Index\n"
        text += "2... "
        text += "Standard deviations\n" if choice == 'alchemite' \
            else "Percentage of Bagging Estimators predicting more extreme value\n"
        text += "3... Input\n"
        text += "4... Prediction\n"
        text += "5... Real value\n"

        risk_metrik = "Standard Deviations" if choice == 'alchemite' \
            else "more extreme bagging estimators"

        text += "{:<10s} {:>5s} {:>16s} {:>16s} {:>16s}\n".format(
            "1", "2", "3", "4", "5"
        )
        for feature in features_display:
            text += "\n\n{}\n".format(feature)

            report_feature = report.loc[report["Column"] == feature,:].copy()
            sort_idx = np.argsort(report_feature["Standard Deviations"])
            print("1 Idx: ")
            print(sort_idx)
            sort_idx = sort_idx[::-1].values
            sort_idx = report_feature["Sample Index"].values[sort_idx]
            print("2 Idx: ")
            print(sort_idx)
            print("Metrik: {}".format(risk_metrik))
            print(report.columns)
            for idx in sort_idx:
                text += "* " if idx in yshuffled_idx and feature in yfeatures else "  "
                print("Idx: {}".format(idx))

                text += "{:<10s}: ".format(str(idx))
                text += "{:>4.3f}".format(report_feature.loc[report_feature["Sample Index"] == idx, risk_metrik].values[0])
                text += "{:>15.3f}".format(report_feature.loc[report_feature["Sample Index"] == idx, "Input"].values[0])
                text += "{:>15.3f}".format(report_feature.loc[report_feature["Sample Index"] == idx, "Prediction"].values[0])
                text += "{:>15.3f}".format(test.loc[idx, feature])
                text += "\n"

        if choice == 'basic':
            text = "Basic\n\n" + text
            outliers['widgets']['yshuffle_info_basic'].delete("1.0", tk.END)
            outliers['values']['yshuffle_info_basic'] = text
            outliers['widgets']['yshuffle_info_basic'].insert(tk.END, outliers['values']['yshuffle_info_basic'])
        elif choice == 'alchemite':
            text = "Alchemite\n\n" + text
            outliers['widgets']['yshuffle_info_alchemite'].delete("1.0", tk.END)
            outliers['values']['yshuffle_info_alchemite'] = text
            outliers['widgets']['yshuffle_info_alchemite'].insert(tk.END, outliers['values']['yshuffle_info_alchemite'])

    print(text)

# FUNCTIONALITY - "Execute" Frame

def loadExecute():
    global analyzer_basic, analyzer_alchemite, test
    text = ""

    execute['widgets']['code'].delete("1.0", tk.END)
    execute['values']['code'] = text
    execute['widgets']['code'].insert(tk.END, execute['values']['code'])


def chooseExecute(choice):
    global analyzer_basic, analyzer_alchemite, test
    if choice == "run":
        code = execute['widgets']['code'].get("1.0", tk.END)
        exec(code)


# WIDGETS DEFINITION

frames = {'menu': tk.LabelFrame(root, width=width, height=50),
    'main': tk.LabelFrame(root, width=width)}
frames.update({
    'data': tk.LabelFrame(frames['main'],
        text="Select data sources",
        width=width, height=(height-50)),
    'explore': tk.LabelFrame(frames['main'],
        text="Explorative analysis",
        width=width, height=(height-50)),
    'alchemite': tk.LabelFrame(frames['main'],
        text=version,
        width=width, height=(height-50)),
    'basic': tk.LabelFrame(frames['main'],
        text="Basic algorithms",
        width=width, height=(height-50)),
    'comparison': tk.LabelFrame(frames['main'],
        text="Comparison",
        width=width, height=(height-50)),
    'outliers': tk.LabelFrame(frames['main'],
        text="Outlier detection",
        width=width, height=(height-50)),
    'execute': tk.LabelFrame(frames['main'],
        text="Execute Code",
        width=width, height=(height-50))
})


data = {
    'values': {
        'train': tk.StringVar(),
        'valid': tk.StringVar(),
        'test': tk.StringVar(),
    }
}
data.update({
    'widgets': {
        'train': tk.Label(frames['data'], text="Train", width=20, anchor='w'),
        'train_filename': tk.Label(frames['data'], textvariable=data['values']['train'], width=50, anchor='w'),
        'train_button': tk.Button(frames['data'], text="Select training data", width=30, command=lambda: chooseData("train")),
        'valid': tk.Label(frames['data'], text="Valid", width=20, anchor='w'),
        'valid_filename': tk.Label(frames['data'], textvariable=data['values']["valid"], width=50, anchor='w'),
        'valid_button': tk.Button(frames['data'], text="Select validation data", width=30, command=lambda: chooseData("valid")),
        'test': tk.Label(frames['data'], text="Test", width=20, anchor='w'),
        'test_filename': tk.Label(frames['data'], textvariable=data['values']["test"], width=50, anchor='w'),
        'test_button': tk.Button(frames['data'], text="Select testing data", width=30, command=lambda: chooseData("test")),
    }
})


explore = {
    'values': {
        'menu': tk.StringVar(),
        'feature': tk.StringVar(),
        'info': "",
    },
    'widgets': {
        'main': tk.LabelFrame(frames['explore']),
    }
}
explore['widgets'].update({
    'dimension_button': tk.Button(frames['explore'], text="Dimensions", width=30, command=lambda: chooseExplore("dimensions")),
    'missing_values_button': tk.Button(frames['explore'], text="Missing Values", width=30, command=lambda: chooseExplore("missing_vals")),
    'basic_stats_button': tk.Button(frames['explore'], text="Basic Stats", width=30, command=lambda: chooseExplore("basic_stats")),
    'densities_button': tk.Button(frames['explore'], text="Densities", width=30, command=lambda: chooseExplore("densities")),
    'feature': tk.OptionMenu(explore['widgets']['main'],
        explore['values']['feature'],
        *(BasicAnalyzer.train.getFeatures() if BasicAnalyzer.train is not None else [""])
    ),
    'info': tk.Text(explore['widgets']['main'])
})


basic = {
    'values': {
        'info': "",
        'feature': tk.StringVar(),
        'show': tk.StringVar(),
    }
}
basic.update({
    'widgets': {
        'train': tk.Button(frames['basic'], text="train Basic", width=30, command=lambda: chooseBasic("train")),
        'train_alchemite': tk.Button(frames['basic'], text="train Alchemite", width=30, command=lambda: chooseBasic("train_alchemite")),
        'save': tk.Button(frames['basic'], text="save feature models", width=30, command=lambda: chooseBasic("save")),
        'load': tk.Button(frames['basic'], text="load feature models", width=30, command=lambda: chooseBasic("load")),
        'feature': tk.OptionMenu(frames['basic'],
            basic['values']['feature'],
            *(BasicAnalyzer.train.getFeatures() if BasicAnalyzer.train is not None else [""])
        ),
        'set_params': tk.Button(frames['basic'], text="set search parameters", width=30, command=lambda: chooseBasic("set_params")),
        'show': tk.OptionMenu(frames['basic'],
            basic['values']['show'],
            *['GCV Paramters', 'Model']
        ),
        'info': tk.Text(frames['basic'], width=160, height=50)
    }
})

comparison = {
    'values': {
        'info_basic': "",
        'info_alchemite': "",
        'iterations': tk.IntVar(),
        'errors': tk.IntVar(),
    }
}
comparison.update({
    'widgets': {
        'iterations_label': tk.Label(frames['comparison'], text="Iterations:", width=20, anchor='w'),
        'iterations': tk.OptionMenu(frames['comparison'],
            comparison['values']['iterations'],
            *[1, 2, 5, 10, 50, 100]
        ),
        'errors_label': tk.Label(frames['comparison'], text="Missing values per sample: ", width=20, anchor='w'),
        'errors': tk.OptionMenu(frames['comparison'],
            comparison['values']['errors'],
            *[1, 2, 3, 4]
        ),
        'compare': tk.Button(frames['comparison'], text="compare methods", width=30, command=lambda: chooseComparison("compare")),
        'export': tk.Button(frames['comparison'], text="export imputated data", width=30, command=lambda: chooseComparison("export")),

        'info_basic': tk.Text(frames["comparison"], width=80, height=50),
        'info_alchemite': tk.Text(frames["comparison"], width=80, height=50),
    }
})

outliers = {
    'values': {
        'distribution_feature': tk.StringVar(),
        'yshuffle_features': {}
    },
    'widgets': {
        'main': tk.LabelFrame(frames['outliers']),
    }
}
outliers['widgets'].update({
    'distribution': tk.LabelFrame(outliers['widgets']['main']),
    'yshuffle': tk.LabelFrame(outliers['widgets']['main']),
    'add3sigma': tk.LabelFrame(outliers['widgets']['main']),
})

outliers['widgets'].update({
    'button_distribution': tk.Button(frames['outliers'], text="Bagging Estimator distribution", width=30, command=lambda: chooseOutliers("distribution")),
    'button_yshuffle': tk.Button(frames['outliers'], text="Y-Shuffled Set", width=30, command=lambda: chooseOutliers("yshuffle")),
    'button_add3sigma': tk.Button(frames['outliers'], text="Add 3 sigma", width=30, command=lambda: chooseOutliers("add3sigma")),
    'distribution_feature': tk.OptionMenu(outliers['widgets']['distribution'],
        outliers['values']['distribution_feature'],
        *(BasicAnalyzer.train.getFeatures() if BasicAnalyzer.train is not None else [""])
    ),
    'yshuffle_menu': tk.LabelFrame(outliers['widgets']['yshuffle']),
    'yshuffle_info': tk.LabelFrame(outliers['widgets']['yshuffle']),
})

outliers['widgets'].update({
    'yshuffle_shuffle': tk.Button(outliers['widgets']['yshuffle_menu'], text="Shuffle", command=lambda: outliers_yshuffle_run("shuffle")),
    'yshuffle_run_basic': tk.Button(outliers['widgets']['yshuffle_menu'], text="Run Basic", command=lambda: outliers_yshuffle_run("basic")),
    'yshuffle_run_alchemite': tk.Button(outliers['widgets']['yshuffle_menu'], text="Run Alchemite", command=lambda: outliers_yshuffle_run("alchemite")),
    'yshuffle_features' : {},
    'yshuffle_info_basic': tk.Text(outliers['widgets']['yshuffle_info'], width=80, height=50),
    'yshuffle_info_alchemite': tk.Text(outliers['widgets']['yshuffle_info'], width=80, height=50),
})



execute = {
    'values': {
        'code': "",
    },
    'widgets': {
        'run': tk.Button(frames['execute'], text="run", width=30, command=lambda: chooseExecute("run")),
        'code': tk.Text(frames['execute'], width=160, height=50),
    }
}



menu = {
    'widgets': {
        'data': tk.Button(frames['menu'], text="Select data", command=lambda: chooseMenu("data")),
        'explore': tk.Button(frames['menu'], text="Explore data", command=lambda: chooseMenu("explore")),
        'basic': tk.Button(frames['menu'], text="Models", command=lambda: chooseMenu("basic")),
        'comparison': tk.Button(frames['menu'], text="Missing Values", command=lambda: chooseMenu("comparison")),
        'outliers': tk.Button(frames['menu'], text="Outliers", command=lambda: chooseMenu("outliers")),
        'execute': tk.Button(frames['menu'], text="Run Code", command=lambda: chooseMenu("execute")),
    }
}


if DEBUG:
    filename = "data\\steel_train.csv"
    data['values']['train'].set(filename)
    df = pd.read_csv(filename, index_col=0).astype('float32')
    train = Dataset("Steel", 'train', df)
    BasicAnalyzer.setTrainDS(train)

    filename = "data\\steel_test.csv"
    data['values']['test'].set(filename)
    df = pd.read_csv(filename, index_col=0).astype('float32')
    test = Dataset("Steel", 'test', df)



# WIDGETS LINKING


# Fills the Menu on the top of the window
for i, button in enumerate(menu['widgets'].values()):
    button.grid(row=0, column=i, sticky='nsew')
    frames['menu'].grid_columnconfigure(i, minsize=20, weight=1)
frames['menu'].grid_rowconfigure(0, minsize=50, weight=1)


# Fills the Data choice
tk.Label(frames['data'], text="Item", anchor="w", font=font_header).grid(row=0, column=0, sticky='nsew')
tk.Label(frames['data'], text="Filename", anchor="w", font=font_header).grid(row=0, column=1, sticky='nsew')
tk.Label(frames['data'], text="Load", anchor="w", font=font_header).grid(row=0, column=2, sticky='nsew')
frames['data'].grid_rowconfigure(0, minsize=20, weight=1)
for i, name in enumerate(['train', 'valid', 'test']):
    data['widgets'][name].grid(row=i+1, column=0, sticky='nsew')
    data['widgets'][name + "_filename"].grid(row=i+1, column=1, sticky='nsew')
    data['widgets'][name + "_button"].grid(row=i+1, column=2, sticky='nsew')
    frames['data'].grid_rowconfigure(i+1, minsize=20, weight=1)
frames['data'].grid_columnconfigure(0, minsize=20, weight=1)
frames['data'].grid_columnconfigure(1, minsize=20, weight=1)
frames['data'].grid_columnconfigure(2, minsize=20, weight=1)


# Fills the Explorative choice
explore['widgets']['dimension_button'].grid(row=0, column=0, sticky='nsew')
explore['widgets']['missing_values_button'].grid(row=0, column=1, sticky='nsew')
explore['widgets']['basic_stats_button'].grid(row=0, column=2, sticky='nsew')
explore['widgets']['densities_button'].grid(row=0, column=3, sticky='nsew')
explore['widgets']['main'].grid(row=1, column=0, columnspan=4, sticky='nsew')
frames['explore'].grid_rowconfigure(0, minsize=50)
frames['explore'].grid_rowconfigure(1, minsize=200, weight=1)
frames['explore'].grid_columnconfigure(0, minsize=200, weight=1)
frames['explore'].grid_columnconfigure(1, minsize=200, weight=1)
frames['explore'].grid_columnconfigure(2, minsize=200, weight=1)
frames['explore'].grid_columnconfigure(3, minsize=200, weight=1)
explore['values']['feature'].trace('w', chooseExplore2)


# Fills the Basic choice
basic['widgets']['train_alchemite'].grid(row=0, column=0, columnspan=3, sticky='nsew')
basic['widgets']['train'].grid(row=1, column=0, sticky='nsew')
basic['widgets']['save'].grid(row=1, column=1, sticky='nsew')
basic['widgets']['load'].grid(row=1, column=2, sticky='nsew')
basic['widgets']['feature'].grid(row=2, column=0, sticky='nsew')
basic['widgets']['set_params'].grid(row=2, column=1, sticky='nsew')
basic['widgets']['show'].grid(row=2, column=2, sticky='nsew')
basic['widgets']['info'].grid(row=3, column=0, columnspan=3, sticky='nsew')
frames['basic'].grid_rowconfigure(0, minsize=50)
frames['basic'].grid_rowconfigure(1, minsize=50)
frames['basic'].grid_rowconfigure(2, minsize=50)
frames['basic'].grid_rowconfigure(3, minsize=200, weight=1)
frames['basic'].grid_columnconfigure(0, minsize=50, weight=1)
frames['basic'].grid_columnconfigure(1, minsize=50, weight=1)
frames['basic'].grid_columnconfigure(2, minsize=50, weight=1)
basic['values']['feature'].trace('w', chooseBasic2)
basic['values']['show'].trace('w', chooseBasic2)


# Fills the Comparison choice
comparison['widgets']['iterations_label'].grid(row=0, column=0, sticky='nsew')
comparison['widgets']['iterations'].grid(row=0, column=1, sticky='nsew')
comparison['widgets']['errors_label'].grid(row=1, column=0, sticky='nsew')
comparison['widgets']['errors'].grid(row=1, column=1, sticky='nsew')
comparison['widgets']['compare'].grid(row=2, column=0, sticky='nsew')
comparison['widgets']['export'].grid(row=2, column=1, sticky='nsew')
comparison['widgets']['info_basic'].grid(row=3, column=0, sticky='nsew')
comparison['widgets']['info_alchemite'].grid(row=3, column=1, sticky='nsew')
frames['comparison'].grid_rowconfigure(0, minsize=50)
frames['comparison'].grid_rowconfigure(1, minsize=50)
frames['comparison'].grid_rowconfigure(2, minsize=50)
frames['comparison'].grid_rowconfigure(3, minsize=200, weight=1)
frames['comparison'].grid_columnconfigure(0, minsize=50, weight=1)
frames['comparison'].grid_columnconfigure(1, minsize=50, weight=1)

# Fills the Outliers frame
outliers['widgets']['button_distribution'].grid(row=0, column=0, sticky='nsew')
outliers['widgets']['button_yshuffle'].grid(row=0, column=1, sticky='nsew')
#outliers['widgets']['button_add3sigma'].grid(row=0, column=2, sticky='nsew')
outliers['widgets']['main'].grid(row=1, column=0, columnspan=3, sticky='nsew')
frames['outliers'].grid_rowconfigure(0, minsize=50)
frames['outliers'].grid_rowconfigure(1, minsize=50, weight=1)
frames['outliers'].grid_columnconfigure(0, minsize=50, weight=1)
frames['outliers'].grid_columnconfigure(1, minsize=50, weight=1)
frames['outliers'].grid_columnconfigure(2, minsize=50, weight=1)
# set distribution as the standard outliers page
outliers['widgets']['distribution'].grid(row=0, column=0, sticky='nsew')
# START: fill the distribution page
outliers['widgets']['distribution_feature'].grid(row=0, column=0, sticky='nsew')
outliers['values']['distribution_feature'].trace('w', chooseOutliers2)
# START: fill the outliers y-shuffle page
outliers['widgets']['yshuffle_menu'].grid(row=0, column=0, sticky='nsew')
outliers['widgets']['yshuffle_info'].grid(row=0, column=1, sticky='nsew')
outliers['widgets']['yshuffle_info_basic'].grid(row=0, column=0, sticky='nsew')
outliers['widgets']['yshuffle_info_alchemite'].grid(row=0, column=1, sticky='nsew')
outliers['widgets']['yshuffle_info'].grid_rowconfigure(0, minsize=100, weight=1)
outliers['widgets']['yshuffle_info'].grid_columnconfigure(0, minsize=100, weight=1)
outliers['widgets']['yshuffle_info'].grid_columnconfigure(1, minsize=100, weight=1)
outliers['widgets']['yshuffle'].grid_rowconfigure(0, minsize=50, weight=1)
outliers['widgets']['yshuffle'].grid_columnconfigure(0, minsize=100)
outliers['widgets']['yshuffle'].grid_columnconfigure(1, minsize=100, weight=1)
# START: configure the outliers main layout
outliers['widgets']['main'].grid_rowconfigure(0, minsize=50, weight=1)
outliers['widgets']['main'].grid_columnconfigure(0, minsize=50, weight=1)


# Fills the Execute choice
execute['widgets']['run'].grid(row=0, column=0, sticky='nsew')
execute['widgets']['code'].grid(row=1, column=0, sticky='nsew')
frames['execute'].grid_rowconfigure(0, minsize=50)
frames['execute'].grid_rowconfigure(1, minsize=50, weight=1)
frames['execute'].grid_columnconfigure(0, minsize=50, weight=1)


# Divide the window into  Menu and Main area
frames['menu'].grid(row=0, column=0, sticky='nsew')
frames['main'].grid(row=1, column=0, sticky='nsew')
root.grid_rowconfigure(0, minsize=50)
root.grid_rowconfigure(1, minsize=200, weight=1)
root.grid_columnconfigure(0, minsize=100, weight=1)

root.mainloop()

analyzer_alchemite.api_models.models_id_delete(analyzer_alchemite.model_id)
analyzer_alchemite.api_datasets.datasets_id_delete(BasicAnalyzer.train.dataset_id)
