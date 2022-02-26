import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from TrainLog import TrainLog
from log import read_log


class LogList(tk.Frame):

    def __init__(self, master, folders):
        super().__init__(master)

        self._folders = []
        self._folder_frames = []
        self.update_folders(folders)

    def get_selected_paths(self):
        paths = []
        for folder in self._folder_frames:
            paths += folder.get_selected_paths()
        return paths

    def update_folders(self, folders):
        self._folders = folders

        for frame in self._folder_frames:
            frame.destroy()

        self._folder_frames = []
        for i in range(len(folders)):
            folder = Folder(self, folders[i][0], folders[i][1])
            self._folder_frames.append(folder)
            folder.grid(row=i, column=0, sticky="we")


class Folder(tk.Frame):

    def __init__(self, master, folder, files):
        super().__init__(master)

        self.config(highlightbackground="black", highlightthickness=1)

        self._files = files
        self._folder = folder

        self._label = tk.Label(self, text=folder)
        self._label.grid(row=0, column=0, sticky="w")

        self._states = [tk.IntVar() for _ in files]
        self._file_buttons = []
        for i in range(len(files)):
            cb = tk.Checkbutton(self, variable=self._states[i], text=files[i])
            self._file_buttons.append(cb)
            cb.grid(row=i + 1, column=0, sticky="w")

    def get_selected_paths(self):
        paths = []
        for i in range(len(self._files)):
            if self._states[i].get():
                paths.append(self._folder + '\\' + self._files[i])
        return paths


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


def draw(fig, canvas, log_list):
    draw_graphics(fig, log_list.get_selected_paths())
    canvas.draw()


def save(fig, path):
    fig.savefig(path)


def update_folders(log_list, folders):
    files = [(folder, os.listdir(folder)) for folder in folders]
    log_list.update_folders(files)


def main():
    folders = ['logs', 'logs_DubinsCar', 'logs_SimpleControlProblem_Discrete']
    files = [(folder, os.listdir(folder)) for folder in folders]

    root = tk.Tk()

    fig = plt.figure(figsize=(16, 8))
    canvas = FigureCanvasTkAgg(fig, root)

    control = tk.Frame(root)
    log_list = LogList(control, files)
    log_list.pack(fill='both')
    draw_button = tk.Button(control, text='Draw', command=lambda: draw(fig, canvas, log_list))
    draw_button.pack()
    save_button = tk.Button(control, text='Update files', command=lambda: update_folders(log_list, folders))
    save_button.pack()
    save_button = tk.Button(control, text='Save', command=lambda: save(fig, 'plots/plot.png'))
    save_button.pack()

    control.pack(side='left')
    canvas.get_tk_widget().pack(side='left', fill=tk.BOTH, expand=1)

    root.mainloop()


if __name__ == '__main__':
    main()
