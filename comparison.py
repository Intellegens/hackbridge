import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText

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
analyzer_1 = None
analyzer_2 = None

analyzers = []

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
    global chosenMenu, frames, height, width, root
    if chosenMenu != None:
        # frames[chosenMenu].pack_forget()
        frames[chosenMenu].grid_forget()
    chosenMenu = choice
    # frames[chosenMenu].pack(fill=tk.BOTH, expand=True)
    if chosenMenu == 'data':
        frames[chosenMenu].grid(row=0, column=0, sticky="n")
    elif chosenMenu != 'exit':
        frames[chosenMenu].grid(row=0, column=0, sticky="nsew")

    frames['main'].grid_columnconfigure(0, minsize=10, weight=1)
    frames['main'].grid_rowconfigure(0, minsize=10, weight=1)

    if choice == "models":
        loadModels()
    elif choice == "missing_vals":
        load_missing_vals()
    elif choice == "outliers":
        loadOutliers()
    elif choice == "execute":
        loadExecute()
    elif choice == "exit":
        root.destroy()


def choose_analyzer_1(*args):
    global analyzers, analyzer_1
    selected_1 = menu["values"]["analyzer_1"].get()
    print("Selction: {}".format(selected_1))
    for analyzer in analyzers:
        if analyzer.name == selected_1:
            analyzer_1 = analyzer
            if hasattr(test, "out_1"):
                delattr(test, "out_1")


def choose_analyzer_2(*args):
    global analyzers, analyzer_2
    selected_2 = menu["values"]["analyzer_2"].get()
    print("Selction: {}".format(selected_2))
    for analyzer in analyzers:
        if analyzer.name == selected_2:
            analyzer_2 = analyzer
            if hasattr(test, "out_2"):
                delattr(test, "out_2")


# FUNCTIONALITY - "Data" Frame

def chooseData(value):
    global filenames
    # filenames.update({value:
    #    filedialog.askopenfilename(initialdir=".", title="Select {}".format(value))})
    filename = filedialog.askopenfilename(initialdir=".", title="Select {}".format(value))
    if filename is not None:
        data['values'][value].set(filename)
        if value == 'train':
            train_df = pd.read_csv(filename, index_col=0).astype('float32')
            print(train_df.head())
            train = Dataset(data['values']['train'].get(), 'train', train_df)
            BasicAnalyzer.setTrainDS(train)
        if value == 'valid':
            valid_df = pd.read_csv(filename, index_col=0).astype('float32')
            print(valid_df.head())
            valid = Dataset(data['values']['valid'].get(), 'valid', valid_df)
            BasicAnalyzer.setValidDS(valid)
        if value == 'test':
            test_df = pd.read_csv(filename, index_col=0).astype('float32')
            print(test_df.head())
            test = Dataset(data['values']['test'].get(), 'test', test_df)
            BasicAnalyzer.setTestDS(test)


# FUNCTIONALITY - "Explore" Frame

state_explore = ''
def chooseExplore(choice):
    print("comparison.chooseExplore")
    global state_explore, canvas_widget, test
    if choice == "densities":
        if state_explore != 'density':
            explore['widgets']['info'].grid_forget()
            explore['widgets']['feature'].grid(row=0, column=0, sticky='nsew')
            explore['widgets']['feature']['menu'].delete(0, 'end')
            features = BasicAnalyzer.train.getFeatures() if BasicAnalyzer.train is not None else []
            for feature in features:
                explore['widgets']['feature']['menu'].add_command(label=feature, command=tk._setit(explore['values']['feature'], feature))
            explore['values']['feature'].set(features[0])
        chooseExplore2()
        state_explore = 'density'
    else:
        if state_explore != 'info':     # show the info text if density is currently displayed
            explore['widgets']['feature'].grid_forget()
            if canvas_widget:
                canvas_widget.destroy()
            explore['widgets']['info'].grid(row=0, column=0, sticky='nsew')
            explore['widgets']['main'].grid_rowconfigure(0, minsize=200, weight=1)
            explore['widgets']['main'].grid_rowconfigure(1, minsize=0, weight=0)
            explore['widgets']['main'].grid_columnconfigure(0, minsize=200, weight=1)
            explore['widgets']['main'].grid_columnconfigure(1, minsize=0, weight=0)
            state_explore = 'info'

        if choice == "dimensions":
            explore['widgets']['info'].delete(1.0, tk.END)
            explore['values']['info'] = BasicAnalyzer.printDimensions(test)
            explore['widgets']['info'].insert(tk.END, explore['values']['info'])
        if choice == "missing_vals":
            explore['widgets']['info'].delete(1.0, tk.END)
            explore['values']['info'] = BasicAnalyzer.printMissingValues(test)
            explore['widgets']['info'].insert(tk.END, explore['values']['info'])
        if choice == "basic_stats":
            explore['widgets']['info'].delete(1.0, tk.END)
            explore['values']['info'] = BasicAnalyzer.printBasicStats(test)
            explore['widgets']['info'].insert(tk.END, explore['values']['info'])

canvas_widget, canvas_widget2 = None, None
def chooseExplore2(*args):
    global test, canvas_widget
    text = ""
    feature = explore['values']['feature'].get()

    if feature:
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



# FUNCTIONALITY - "Models" Frame

def loadModels():
    print("comparison.loadModels")

    models['values']['show'].set("GCV Parameters")

    models['widgets']['feature']['menu'].delete(0, 'end')
    models['widgets']['feature']['menu'].add_command(label="", command=tk._setit(models['values']['feature'], ""))
    for feature in (BasicAnalyzer.train.getFeatures() if BasicAnalyzer.train is not None else []):
        models['widgets']['feature']['menu'].add_command(label=feature, command=tk._setit(models['values']['feature'], feature))

    update_models_analyzer()


analyzer_models = None
def chooseModels(choice):
    global analyzers, analyzer_models
    text = ""
    model_name = models['widgets']['analyzer_name'].get()
    print("Model Name: {}".format(model_name))

    if choice == 'basic_add':
        analyzer_models = BasicAnalyzer(model_name)

        text = ""
        text += BasicAnalyzer.params_std
        models['widgets']['info'].delete("1.0", tk.END)
        models['values']['info'] = text
        models['widgets']['info'].insert(tk.END, models['values']['info'])

        analyzers.append(analyzer_models)
        update_models_analyzer()

    elif choice == 'alchemite_add':
        analyzer_models = Alchemite.AlchemiteAnalyzer(model_name,
            credentials=filedialog.askopenfilename(initialdir=".", title="Credentials"))
        analyzers.append(analyzer_models)
        update_models_analyzer()

    elif analyzer_models is not None:
        analyzer_type = analyzer_models.__class__.__name__
        
        if choice == 'show_residuals':
             text = analyzer_models.print_residuals(test)
             models['widgets']['info'].delete("1.0", tk.END)
             models['values']['info'] = text
             models['widgets']['info'].insert(tk.END, models['values']['info'])
            
        elif analyzer_type == 'AlchemiteAnalyzer':
            if choice == 'train_alchemite':
                analyzer_models.fitFeatureModels()
            else:
                print("This operation cannot be perfomed with an Alchemite Model")

        elif analyzer_type == 'BasicAnalyzer':
            if choice == 'set_params':
                feature = models['values']['feature'].get()
                if feature:
                    param_code = models['widgets']['info'].get(1.0,'end')
                    analyzer_models.gcvs["params"].update({feature: param_code})

            elif choice == 'set_std_params':
                param_code = models['widgets']['info'].get(1.0,'end')
                analyzer_models.gcvs.update({"params_std": param_code})

            elif choice == 'train_basic':
                BasicAnalyzer.train.fillMissingValues(np.average)
                analyzer_models.fitFeatureModels()
                BasicAnalyzer.train.restoreMissingValues()

            elif choice == 'load_basic':
                filename = filedialog.askopenfilename(initialdir=".", title="Load {}".format("Model"))
                analyzer_models.loadGCV(filename)

            elif choice == 'save_basic':
                filename = filedialog.asksaveasfilename(initialdir=".", title="Save {}".format("Model"))
                analyzer_models.saveGCV(filename)
            
            elif choice == 'show_std_params':
                text = analyzer_models.gcvs['params_std']
                models['widgets']['info'].delete("1.0", tk.END)
                models['values']['info'] = text
                models['widgets']['info'].insert(tk.END, models['values']['info'])

            else:
                print("This operation cannot be performed with a Basic Model")

        else:
            print("Unknown Analyzer Class")

    else:
        print("No Analyzer added Yet")


def chooseModels2(*args):
    global analyzer_models
    text = ""

    if analyzer_models is not None:
        analyzer_type = analyzer_models.__class__.__name__
        if analyzer_type == "BasicAnalyzer":
            feature = models['values']['feature'].get()
            show = models['values']['show'].get()

            if show != "Model":
                text += analyzer_models.gcvs["params"][feature] if feature in analyzer_models.gcvs["params"] else analyzer_models.gcvs["params_std"]
            else:
                text += analyzer_models.getFeatureModelDescription(feature) if feature in analyzer_models.gcvs["models"] else ""

    models['widgets']['info'].delete(1.0, tk.END)
    models['values']['info'] = text
    models['widgets']['info'].insert(tk.END, models['values']['info'])


def choose_models_analyzer(*args):
    global analyzer_models
    selected = models['values']['analyzer'].get()
    for analyzer in analyzers:
        if analyzer.name == selected:
            analyzer_models = analyzer
            models['widgets']['analyzer_name'].delete("0", tk.END)
            models['widgets']['analyzer_name'].insert(tk.END, selected)

            chooseModels2()


def update_models_analyzer():
    menu['widgets']['analyzer_1']['menu'].delete(0, 'end')
    menu['widgets']['analyzer_2']['menu'].delete(0, 'end')
    for analyzer in analyzers:
        menu['widgets']['analyzer_1']['menu'].add_command(label=analyzer.name, command=tk._setit(menu['values']['analyzer_1'], analyzer.name))
        menu['widgets']['analyzer_2']['menu'].add_command(label=analyzer.name, command=tk._setit(menu['values']['analyzer_2'], analyzer.name))

    
    models['widgets']['analyzer']['menu'].delete(0, 'end')
    for analyzer in analyzers:
        models['widgets']['analyzer']['menu'].add_command(label=analyzer.name, command=tk._setit(models['values']['analyzer'], analyzer.name))



# FUNCTIONALITY - "Missing Values" Frame

def load_missing_vals():
    pass


test_basic_imp, test_alchemite_imp = None, None
def choose_missing_vals(choice):
    global analyzer_1, analyzer_2, test, test_basic_imp, test_alchemite_imp

    if choice == 'create_testset':
        test.getImputationTestset(missing_vals['values']['errors'].get())
        test_basic_imp = Dataset(test.name, "basic_imp",
            test.imp_test_set.copy(deep=True))
        test_alchemite_imp = Dataset(test.name, "alchemite_imp",
            test.imp_test_set.copy(deep=True))

    elif choice[0:6] == 'impute':
        analyzer = None
        text = ""
        if choice[7:] == 'basic':
            analyzer_1.iterateMissingValuePredictions(test_basic_imp,
                iterations=missing_vals['values']['iterations'].get())
            text += "Basic model after {} iterations\n".format(BasicAnalyzer.iterations)
            text += analyzer_1.printImputedVsActual(test, test_basic_imp)
            choice = 'basic'
            analyzer = analyzer_1

        elif choice[7:] == 'alchemite':
            analyzer_2.iterateMissingValuePredictions(test_alchemite_imp)
            text += "Alchemite\n"
            text += analyzer_2.printImputedVsActual(test, test_alchemite_imp)
            choice = 'alchemite'
            analyzer = analyzer_2

        text += "\n\nSetting Column to None"
        features = test.getFeatures()
        for feature in features:
            test.getImputationTestset(1, [feature])
            test_imp = Dataset(test.name, "{}_imp_{}".format(choice, feature),
                test.imp_test_set.copy(deep=True))

            if choice == 'basic':
                analyzer.iterateMissingValuePredictions(
                    test_imp,
                    iterations=missing_vals['values']['iterations'].get(),
                    features=[feature])
            else:
                analyzer.iterateMissingValuePredictions(test_imp)

            text += analyzer.printImputedVsActual(test, test_imp, [feature], False)

        missing_vals['widgets']['info_{}'.format(choice)].delete(1.0, tk.END)
        missing_vals['values']['info_{}'.format(choice)] = text
        missing_vals['widgets']['info_{}'.format(choice)].insert(tk.END, missing_vals['values']['info_{}'.format(choice)])

    elif choice[0:6] == 'export':
        if choice[7:] == 'basic':
            if analyzer_1.imputed is not None:
                filename = filedialog.asksaveasfilename(initialdir=".", title="Export Basic")
                analyzer_1.saveImputedAlchemiteFormat(filename)
        elif choice[7:] == 'alchemite':
            if analyzer_2.imputed is not None:
                filename = filedialog.asksaveasfilename(initialdir=".", title="Export Alchemite")
                analyzer_2.saveImputedAlchemiteFormat(filename)

# FUNCTIONALITY - "Outlier" Frame

def loadOutliers():
    print("comparison.loadOutliers")
    global test

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


def chooseOutliers2(*args):
    print("comparison.chooseOutliers2")

    global analyzer_1, test, canvas_widget, canvas_widget2
    feature = outliers['values']['distribution_feature'].get()

    if feature and analyzer_1 and analyzer_2:
        if not hasattr(test, "out_1"):
            _, _, test.out_1, _ = analyzer_1.findOutliers(test)
        if not hasattr(test, "out_2"):
            _, _, test.out_2, _ = analyzer_2.findOutliers(test)

        fig = Dataset.plotDensities([test.out_1], features=[feature], x_range=[0.0, 1.0])
        if canvas_widget:
            canvas_widget.destroy()
        canvas = FigureCanvasTkAgg(fig, outliers['widgets']['distribution'])
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=2, column=0, sticky='nsew')

        fig2 = Dataset.plotDensities([test.out_2], features=[feature], x_range=[0.0, 1.0])
        if canvas_widget2:
            canvas_widget2.destroy()
        canvas2 = FigureCanvasTkAgg(fig2, outliers['widgets']['distribution'])
        canvas_widget2 = canvas2.get_tk_widget()
        canvas_widget2.grid(row=2, column=1, sticky='nsew')


        outliers['widgets']['distribution'].grid_rowconfigure(0, minsize=50)
        outliers['widgets']['distribution'].grid_rowconfigure(1, minsize=50)
        outliers['widgets']['distribution'].grid_rowconfigure(2, minsize=50, weight=1)
        outliers['widgets']['distribution'].grid_columnconfigure(0, minsize=50, weight=1)
        outliers['widgets']['distribution'].grid_columnconfigure(1, minsize=50, weight=1)


yshuffled_idx, yshuffled = None, None
def outliers_yshuffle_run(choice):
    print("comparison.outliers_yshuffle_run")
    global analyzer_1, analyzer_2, test, yshuffled_idx, yshuffled
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

    if choice == 'alchemite' or choice == 'basic':
        if choice == 'alchemite':
            analyzer = analyzer_2
        elif choice == 'basic':
            analyzer = analyzer_1

        estimated, estimated_std, estimated_pc, report = analyzer.findOutliers(yshuffled)

        text += "1... Index\n"
        text += "2... "
        text += "Standard deviations\n" if choice == 'alchemite' \
            else "Percentage of Bagging Estimators predicting more extreme value\n"
        text += "3... Input\n"
        text += "4... Prediction\n"
        text += "5... Real value\n"

        risk_metrik, risk_sort = ("Standard Deviations", False) if choice == 'alchemite' \
            else ("more extreme bagging estimators", True)

        text += "{:<10s} {:>5s} {:>16s} {:>16s} {:>16s}\n".format(
            "1", "2", "3", "4", "5"
        )
        for feature in features_display:
            text += "\n\n{}\n".format(feature)

            report_feature = report.loc[report["Column"] == feature,:].copy()

            report_feature = report_feature.sort_values(
                by=[risk_metrik, "Standard Deviations"],
                ascending=[risk_sort, False]
            )

            #sort_idx = np.argsort(report_feature["Standard Deviations"])
            #sort_idx = sort_idx[::-1].values
            #sort_idx = report_feature["Sample Index"].values[sort_idx]

            for idx in report_feature["Sample Index"]:
                text += "* " if idx in yshuffled_idx and feature in yfeatures else "  "

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
    global analyzer_1, analyzer_2, test
    text = ""

    execute['widgets']['code'].delete("1.0", tk.END)
    execute['values']['code'] = text
    execute['widgets']['code'].insert(tk.END, execute['values']['code'])


def chooseExecute(choice):
    global analyzer_1, analyzer_2, test
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
    'models': tk.LabelFrame(frames['main'],
        text="Models",
        width=width, height=(height-50)),
    'missing_vals': tk.LabelFrame(frames['main'],
        text="Missing values",
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
    'info': ScrolledText(explore['widgets']['main'])
})


models = {
    'values': {
        'info': "",
        'feature': tk.StringVar(),
        'show': tk.StringVar(),
        'analyzer': tk.StringVar(),
    }
}
models.update({
    'widgets': {
        'analyzer_name': tk.Entry(frames['models'], width=50),
        'basic_add': tk.Button(frames['models'], text="add Basic", width=30, command=lambda: chooseModels("basic_add")),
        'alchemite_add': tk.Button(frames['models'], text="add Alchemite", width=30, command=lambda: chooseModels("alchemite_add")),
        'analyzer': tk.OptionMenu(frames['models'],
            models['values']['analyzer'],
            [x.name for x in analyzers]
        ),
        'train_basic': tk.Button(frames['models'], text="train Basic feature models", width=30, command=lambda: chooseModels("train_basic")),
        'save_basic': tk.Button(frames['models'], text="save Basic feature models", width=30, command=lambda: chooseModels("save_basic")),
        'load_basic': tk.Button(frames['models'], text="load Basic feature models", width=30, command=lambda: chooseModels("load_basic")),
        'train_alchemite': tk.Button(frames['models'], text="train Alchemite", width=30, command=lambda: chooseModels("train_alchemite")),
        'show': tk.OptionMenu(frames['models'],
            models['values']['show'],
            *['GCV Paramters', 'Model']
        ),
        'feature': tk.OptionMenu(frames['models'],
            models['values']['feature'],
            *(BasicAnalyzer.train.getFeatures() if BasicAnalyzer.train is not None else [""])
        ),
        'set_std_params': tk.Button(frames['models'], text="set standard search parameters", width=30, command=lambda: chooseModels("set_std_params")),
        'set_params': tk.Button(frames['models'], text="set feature search parameters", width=30, command=lambda: chooseModels("set_params")),
        'info': ScrolledText(frames['models'], width=160, height=50),
        'show_residuals': tk.Button(frames['models'], text="show feature models' residuals", width=30, command=lambda: chooseModels("show_residuals")),
        'show_std_params': tk.Button(frames['models'], text="show feature models' standard parameters", width=30, command=lambda: chooseModels("show_std_params")),
    }
})

missing_vals = {
    'values': {
        'info_basic': "",
        'info_alchemite': "",
        'iterations': tk.IntVar(),
        'errors': tk.IntVar(),
    }
}
missing_vals.update({
    'widgets': {
        'iterations_label': tk.Label(frames['missing_vals'], text="Iterations:", width=20, anchor='w'),
        'iterations': tk.OptionMenu(frames['missing_vals'],
            missing_vals['values']['iterations'],
            *[1, 2, 5, 10, 50, 100]
        ),
        'errors_label': tk.Label(frames['missing_vals'], text="Missing values per sample: ", width=20, anchor='w'),
        'errors': tk.OptionMenu(frames['missing_vals'],
            missing_vals['values']['errors'],
            *[1, 2, 3, 4]
        ),
        'create_testset': tk.Button(frames['missing_vals'], text="create testset", width=30, command=lambda: choose_missing_vals("create_testset")),
        'impute_basic': tk.Button(frames['missing_vals'], text="impute with Basic", width=30, command=lambda: choose_missing_vals("impute_basic")),
        'impute_alchemite': tk.Button(frames['missing_vals'], text="impute with Alchemite", width=30, command=lambda: choose_missing_vals("impute_alchemite")),
        'export_basic': tk.Button(frames['missing_vals'], text="export imputated Basic data", width=30, command=lambda: choose_missing_vals("export_basic")),
        'export_alchemite': tk.Button(frames['missing_vals'], text="export imputated Alchemite data", width=30, command=lambda: choose_missing_vals("export_alchemite")),
        'info_basic': ScrolledText(frames["missing_vals"], width=80, height=50),
        'info_alchemite': ScrolledText(frames["missing_vals"], width=80, height=50),
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
})

outliers['widgets'].update({
    'button_distribution': tk.Button(frames['outliers'], text="Bagging Estimator distribution", width=30, command=lambda: chooseOutliers("distribution")),
    'button_yshuffle': tk.Button(frames['outliers'], text="Y-Shuffled Set", width=30, command=lambda: chooseOutliers("yshuffle")),
    'distribution_feature': tk.OptionMenu(outliers['widgets']['distribution'],
        outliers['values']['distribution_feature'],
        *(BasicAnalyzer.train.getFeatures() if BasicAnalyzer.train is not None else [""])
    ),
    'distribution_basic': tk.Label(outliers['widgets']['distribution'], text="Basic", width=20, anchor='c'),
    'distribution_alchemite': tk.Label(outliers['widgets']['distribution'], text="Alchemite", width=20, anchor='c'),
    'yshuffle_menu': tk.LabelFrame(outliers['widgets']['yshuffle']),
    'yshuffle_info': tk.LabelFrame(outliers['widgets']['yshuffle']),
})

outliers['widgets'].update({
    'yshuffle_shuffle': tk.Button(outliers['widgets']['yshuffle_menu'], text="Shuffle", command=lambda: outliers_yshuffle_run("shuffle")),
    'yshuffle_run_basic': tk.Button(outliers['widgets']['yshuffle_menu'], text="Run Basic", command=lambda: outliers_yshuffle_run("basic")),
    'yshuffle_run_alchemite': tk.Button(outliers['widgets']['yshuffle_menu'], text="Run Alchemite", command=lambda: outliers_yshuffle_run("alchemite")),
    'yshuffle_features' : {},
    'yshuffle_info_basic': ScrolledText(outliers['widgets']['yshuffle_info'], width=80, height=50),
    'yshuffle_info_alchemite': ScrolledText(outliers['widgets']['yshuffle_info'], width=80, height=50),
})



execute = {
    'values': {
        'code': "",
    },
    'widgets': {
        'run': tk.Button(frames['execute'], text="run", width=30, command=lambda: chooseExecute("run")),
        'code': ScrolledText(frames['execute'], width=160, height=50),
    }
}



menu = {
    'values': {
        'analyzer_1': tk.StringVar(),
        'analyzer_2': tk.StringVar(),
    }
}
menu.update({
    'widgets': {
        'data': tk.Button(frames['menu'], text="Select data", command=lambda: chooseMenu("data")),
        'explore': tk.Button(frames['menu'], text="Explore data", command=lambda: chooseMenu("explore")),
        'models': tk.Button(frames['menu'], text="Models", command=lambda: chooseMenu("models")),
        'missing_vals': tk.Button(frames['menu'], text="Missing Values", command=lambda: chooseMenu("missing_vals")),
        'outliers': tk.Button(frames['menu'], text="Outliers", command=lambda: chooseMenu("outliers")),
        'execute': tk.Button(frames['menu'], text="Run Code", command=lambda: chooseMenu("execute")),

        'analyzer_1': tk.OptionMenu(frames['menu'],
            menu['values']['analyzer_1'],
            [x.name for x in analyzers]
        ),
        'analyzer_2': tk.OptionMenu(frames['menu'],
            menu['values']['analyzer_2'],
            [x.name for x in analyzers]
        ),
        # 'exit': tk.Button(frames['menu'], text="Exit", command=lambda: chooseMenu("exit")),
    }
})


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
menu['widgets']['data'].grid(row=0, column=0, sticky='nsew')
menu['widgets']['explore'].grid(row=0, column=1, sticky='nsew')
menu['widgets']['models'].grid(row=0, column=2, sticky='nsew')
menu['widgets']['missing_vals'].grid(row=0, column=3, sticky='nsew')
menu['widgets']['outliers'].grid(row=0, column=4, sticky='nsew')
menu['widgets']['execute'].grid(row=0, column=5, sticky='nsew')
menu['widgets']['analyzer_1'].grid(row=1, column=0, columnspan=3, sticky='nsew')
menu['widgets']['analyzer_2'].grid(row=1, column=3, columnspan=3, sticky='nsew')
frames['menu'].grid_columnconfigure(0, minsize=20, weight=1)
frames['menu'].grid_columnconfigure(1, minsize=20, weight=1)
frames['menu'].grid_columnconfigure(2, minsize=20, weight=1)
frames['menu'].grid_columnconfigure(3, minsize=20, weight=1)
frames['menu'].grid_columnconfigure(4, minsize=20, weight=1)
frames['menu'].grid_columnconfigure(5, minsize=20, weight=1)
frames['menu'].grid_rowconfigure(0, minsize=50, weight=1)
frames['menu'].grid_rowconfigure(1, minsize=50, weight=1)
menu['values']['analyzer_1'].trace('w', choose_analyzer_1)
menu['values']['analyzer_2'].trace('w', choose_analyzer_2)



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


# Fills the Models choice
models['widgets']['analyzer_name'].grid(row=0, column=0, columnspan=3, sticky='nsew')
models['widgets']['basic_add'].grid(row=1, column=0, columnspan=2, sticky='nsew')
models['widgets']['analyzer'].grid(row=2, column=0, columnspan=3, sticky='nsew')
models['widgets']['train_basic'].grid(row=3, column=0, columnspan=2, sticky='nsew')
models['widgets']['load_basic'].grid(row=4, column=0, sticky='nsew')
models['widgets']['save_basic'].grid(row=4, column=1, sticky='nsew')
models['widgets']['show'].grid(row=5, column=0, sticky='nsew')
models['widgets']['feature'].grid(row=5, column=1, sticky='nsew')
models['widgets']['set_std_params'].grid(row=6, column=0, sticky='nsew')
models['widgets']['set_params'].grid(row=6, column=1, sticky='nsew')
models['widgets']['info'].grid(row=7, column=0, columnspan=3, sticky='nsew')
models['widgets']['show_residuals'].grid(row=8, column=0, columnspan=3, sticky='nsew')
models['widgets']['show_std_params'].grid(row=9, column=0, columnspan=3, sticky='nsew')
models['widgets']['alchemite_add'].grid(row=1, column=2, sticky='nsew')
models['widgets']['train_alchemite'].grid(row=3, column=2, rowspan=4, sticky='nsew')
frames['models'].grid_rowconfigure(0, minsize=50)
frames['models'].grid_rowconfigure(1, minsize=50)
frames['models'].grid_rowconfigure(2, minsize=50)
frames['models'].grid_rowconfigure(3, minsize=50)
frames['models'].grid_rowconfigure(4, minsize=50)
frames['models'].grid_rowconfigure(5, minsize=50)
frames['models'].grid_rowconfigure(6, minsize=50)
frames['models'].grid_rowconfigure(7, minsize=200, weight=1)
frames['models'].grid_rowconfigure(8, minsize=50)
frames['models'].grid_rowconfigure(9, minsize=50)
frames['models'].grid_columnconfigure(0, minsize=50, weight=1)
frames['models'].grid_columnconfigure(1, minsize=50, weight=1)
frames['models'].grid_columnconfigure(2, minsize=100, weight=2)
models['values']['feature'].trace('w', chooseModels2)
models['values']['show'].trace('w', chooseModels2)
models['values']['analyzer'].trace('w', choose_models_analyzer)


# Fills the Missing Values choice
missing_vals['widgets']['iterations_label'].grid(row=0, column=0, sticky='nsew')
missing_vals['widgets']['iterations'].grid(row=0, column=1, sticky='nsew')
missing_vals['widgets']['errors_label'].grid(row=1, column=0, sticky='nsew')
missing_vals['widgets']['errors'].grid(row=1, column=1, sticky='nsew')
missing_vals['widgets']['create_testset'].grid(row=2, column=0, columnspan=2, sticky='nsew')
missing_vals['widgets']['impute_basic'].grid(row=3, column=0, sticky='nsew')
missing_vals['widgets']['impute_alchemite'].grid(row=3, column=1, sticky='nsew')
missing_vals['widgets']['export_basic'].grid(row=4, column=0, sticky='nsew')
missing_vals['widgets']['export_alchemite'].grid(row=4, column=1, sticky='nsew')
missing_vals['widgets']['info_basic'].grid(row=5, column=0, sticky='nsew')
missing_vals['widgets']['info_alchemite'].grid(row=5, column=1, sticky='nsew')
frames['missing_vals'].grid_rowconfigure(0, minsize=50)
frames['missing_vals'].grid_rowconfigure(1, minsize=50)
frames['missing_vals'].grid_rowconfigure(2, minsize=50)
frames['missing_vals'].grid_rowconfigure(3, minsize=50)
frames['missing_vals'].grid_rowconfigure(4, minsize=50)
frames['missing_vals'].grid_rowconfigure(5, minsize=200, weight=1)
frames['missing_vals'].grid_columnconfigure(0, minsize=50, weight=1)
frames['missing_vals'].grid_columnconfigure(1, minsize=50, weight=1)



# Fills the Outliers frame
outliers['widgets']['button_distribution'].grid(row=0, column=0, sticky='nsew')
outliers['widgets']['button_yshuffle'].grid(row=0, column=1, sticky='nsew')
outliers['widgets']['main'].grid(row=1, column=0, columnspan=2, sticky='nsew')
frames['outliers'].grid_rowconfigure(0, minsize=50)
frames['outliers'].grid_rowconfigure(1, minsize=50, weight=1)
frames['outliers'].grid_columnconfigure(0, minsize=50, weight=1)
frames['outliers'].grid_columnconfigure(1, minsize=50, weight=1)
# set distribution as the standard outliers page
outliers['widgets']['distribution'].grid(row=0, column=0, sticky='nsew')
# START: fill the distribution page
outliers['widgets']['distribution_feature'].grid(row=0, column=0, columnspan=2, sticky='nsew')
outliers['values']['distribution_feature'].trace('w', chooseOutliers2)
outliers['widgets']['distribution_basic'].grid(row=1, column=0, sticky='nsew')
outliers['widgets']['distribution_alchemite'].grid(row=1, column=1, sticky='nsew')
outliers['widgets']['distribution'].grid_rowconfigure(0, minsize=50)
outliers['widgets']['distribution'].grid_rowconfigure(1, minsize=50)
outliers['widgets']['distribution'].grid_rowconfigure(2, minsize=50, weight=1)
outliers['widgets']['distribution'].grid_columnconfigure(0, minsize=50, weight=1)
outliers['widgets']['distribution'].grid_columnconfigure(1, minsize=50, weight=1)
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
root.grid_rowconfigure(0, minsize=100)
root.grid_rowconfigure(1, minsize=200, weight=1)
root.grid_columnconfigure(0, minsize=100, weight=1)

# root.overrideredirect(True)
root.mainloop()

for analyzer in analyzers:
    analyzer.close()
