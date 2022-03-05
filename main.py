import tkinter as tk
from tkinter import ttk

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from train_info.train_log import TrainLog
from log import read_log


class ScrollableFrame(ttk.Frame):
    def __init__(self, container, height, width, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self, height=height, width=width)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


class TrainInfo:

    def __init__(self, name, hyper_parameters, epoch_count, file):
        self.name = name
        self.hyper_parameters = hyper_parameters
        self.epoch_count = epoch_count
        self.file = file


def train_log_to_info(train_log: TrainLog, file):
    return TrainInfo(train_log.name, train_log.hyper_parameters, len(train_log.epochs), file)


class FileInfo(tk.Frame):

    def __init__(self, master, variable, train_info: TrainInfo):
        super().__init__(master)

        self._variable = variable
        self._train_info = train_info

        control_frame = tk.Frame(self)
        control_frame.pack(fill='x')

        info_button = tk.Button(control_frame, text='\\/', command=self.expand_info)
        info_button.pack(side='left')

        check_button = tk.Checkbutton(control_frame, variable=self._variable, text=self._train_info.name)
        check_button.pack(side='left')

        self._train_info_frame = None

    def expand_info(self):
        if self._train_info_frame:
            self._train_info_frame.destroy()
            self._train_info_frame = None
        else:
            info = tk.Frame(self)
            info.pack()
            self._train_info_frame = info

            name = tk.Label(info, text='Name: ' + self._train_info.name)
            name.pack(anchor='w')
            file = tk.Label(info, text='File: ' + self._train_info.file)
            file.pack(anchor='w')
            hyper_parameter = tk.Label(info, text='Hyper parameters:')
            hyper_parameter.pack(anchor='w')
            for key in self._train_info.hyper_parameters:
                parameter = tk.Label(info, text=key + ': ' + str(self._train_info.hyper_parameters[key]))
                parameter.pack(anchor='w')


class FolderFrame(tk.Frame):

    def __init__(self, master, folder, fig, call_back):
        super().__init__(master)

        self._fig = fig
        self._canvas = FigureCanvasTkAgg(fig, self)

        self._folder = folder

        self._control_frame = tk.Frame(self)
        self._control_frame.grid(column=0, row=0, sticky="n")

        self._file_frame = ScrollableFrame(self._control_frame, 700, 300)
        self._file_frame.pack(side='top')

        self._label = tk.Label(self._file_frame.scrollable_frame, text=self._folder)
        self._label.grid(row=0, column=0, sticky="w")

        self._file_info_list = []
        self._update_file_frame()

        self._buttons_frame = tk.Frame(self._control_frame)
        self._buttons_frame.pack(side='top', fill='x', padx=1, pady=1)

        draw_button = tk.Button(self._buttons_frame, text='Draw',
                                command=lambda: draw(fig, self._canvas, self._get_selected_paths()))
        draw_button.pack(fill='x', padx=1, pady=1)
        update_button = tk.Button(self._buttons_frame, text='Update files', command=lambda: self._update_file_frame())
        update_button.pack(fill='x', padx=1, pady=1)
        save_button = tk.Button(self._buttons_frame, text='Save', command=lambda: save(None, 'plots/plot.png'))
        save_button.pack(fill='x', padx=1, pady=1)
        back_button = tk.Button(self._buttons_frame, text='Back', command=call_back)
        back_button.pack(fill='x', padx=1, pady=1)

        self._canvas_widget = self._canvas.get_tk_widget()
        self._canvas_widget.grid(column=1, row=0)

    def _get_selected_paths(self):
        paths = []
        for i in range(len(self._files)):
            if self._states[i].get():
                paths.append(self._folder + '\\' + self._files[i].file)
        return paths

    def _update_files(self):
        folder = self._folder
        self._files = [train_log_to_info(TrainLog.from_dict(read_log(folder + '\\' + file)), file)
                       for file in os.listdir(folder)]

    def _update_file_frame(self):
        self._update_files()

        for button in self._file_info_list:
            button.destroy()

        self._states = [tk.IntVar() for _ in self._files]
        for i in range(len(self._files)):
            file_info = FileInfo(self._file_frame.scrollable_frame, self._states[i], self._files[i])
            self._file_info_list.append(file_info)
            file_info.grid(row=i + 1, column=0, sticky="w")


class App(tk.Frame):

    def __init__(self, master, folders):
        super().__init__(master)

        self._folders = folders

        self._fig = plt.figure(figsize=(16, 8))

        self._set_folder_list()

    def _set_folder_list(self):
        self._folder_list = tk.Frame(self)
        for folder in self._folders:
            button = tk.Button(self._folder_list, text=folder, command=lambda f=folder: self._open_folder(f))
            button.pack(fill='x', padx=1, pady=1)
        self._folder_list.pack()

    def _open_folder(self, folder):
        self._folder_list.destroy()
        self._folder_frame = FolderFrame(self, folder, self._fig, self._back)
        self._folder_frame.pack()

    def _back(self):
        self._folder_frame.destroy()
        self._set_folder_list()


def draw_graphics(fig, files):
    log_data = [read_log(path) for path in files]

    fig.clf()

    train_mean_ax = fig.add_subplot(221)
    test_mean_ax = fig.add_subplot(222)
    train_reward_ax = fig.add_subplot(223)
    test_reward_ax = fig.add_subplot(224)

    for data in log_data:
        train_info = TrainLog.from_dict(data)
        time = [0]
        for epoch in train_info.epochs:
            time.append(time[-1] + epoch.time)

        train_mean_reward = [0] + [epoch.train_mean for epoch in train_info.epochs]
        test_mean_reward = [0] + [epoch.test_mean for epoch in train_info.epochs]
        train_mean_ax.plot(time, train_mean_reward, label=train_info.name)
        test_mean_ax.plot(time, test_mean_reward, label=train_info.name)

        test_rewards = [0] + [a for epoch in train_info.epochs for a in epoch.test_rewards]
        time = [0]
        for epoch in train_info.epochs:
            dt = epoch.time / len(epoch.test_rewards)
            shift = time[-1]
            epoch_rewards_times = [shift + dt * (t + 1) for t in range(len(epoch.test_rewards))]
            time += epoch_rewards_times
        test_reward_ax.plot(time, test_rewards, label=train_info.name)

        train_rewards = [0] + [a for epoch in train_info.epochs for a in epoch.train_rewards]
        time = [0]
        for epoch in train_info.epochs:
            dt = epoch.time / len(epoch.train_rewards)
            shift = time[-1]
            epoch_rewards_times = [shift + dt * (t + 1) for t in range(len(epoch.train_rewards))]
            time += epoch_rewards_times
        train_reward_ax.plot(time, train_rewards, label=train_info.name)

    for ax in [train_mean_ax, test_mean_ax, train_reward_ax, test_reward_ax]:
        ax.set_ylabel('reward')
        ax.set_xlabel('time')
        ax.legend()

    train_mean_ax.title.set_text('Train mean')
    test_mean_ax.title.set_text('Test mean')
    train_reward_ax.title.set_text('Train rewards')
    test_reward_ax.title.set_text('Test rewards')


def draw(fig, canvas, files):
    draw_graphics(fig, files)
    canvas.draw()


def save(fig, path):
    fig.savefig(path)


def update_folders(log_list, folders):
    files = [(folder, os.listdir(folder)) for folder in folders]
    log_list.update_folders(files)


def main():
    folders = ['logs', 'logs_DubinsCar', 'logs_SimpleControlProblem_Discrete']

    root = tk.Tk()

    app = App(root, folders)
    app.pack()

    root.mainloop()


if __name__ == '__main__':
    main()
