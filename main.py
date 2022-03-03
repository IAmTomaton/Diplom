import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from TrainLog import TrainLog
from log import read_log


class FolderFrame(tk.Frame):

    def __init__(self, master, folder, fig, call_back):
        super().__init__(master)

        self._fig = fig
        self._canvas = FigureCanvasTkAgg(fig, self)

        self._files = os.listdir(folder)
        self._folder = folder

        self._control_frame = tk.Frame(self)
        self._control_frame.grid(column=0, row=0, sticky="n")

        self._file_frame = tk.Frame(self._control_frame)
        self._file_frame.pack(side='top')

        self._label = tk.Label(self._file_frame, text=self._folder)
        self._label.grid(row=0, column=0, sticky="w")

        self._file_buttons = []
        self._update_file_frame()

        self._buttons_frame = tk.Frame(self._control_frame)
        self._buttons_frame.pack(side='top', fill='x', padx=1, pady=1)

        draw_button = tk.Button(self._buttons_frame, text='Draw',
                                command=lambda: draw(fig, self._canvas, self._get_selected_paths()))
        draw_button.pack(fill='x', padx=1, pady=1)
        update_button = tk.Button(self._buttons_frame, text='Update files', command=lambda: self._update_files())
        update_button.pack(fill='x', padx=1, pady=1)
        save_button = tk.Button(self._buttons_frame, text='Save', command=lambda: save(None, 'plots/plot.png'))
        save_button.pack(fill='x', padx=1, pady=1)
        back_button = tk.Button(self._buttons_frame, text='Back', command=call_back)
        back_button.pack(fill='x', padx=1, pady=1)

        self._canvas_widget = self._canvas.get_tk_widget()
        self._canvas_widget .grid(column=1, row=0)

    def _get_selected_paths(self):
        paths = []
        for i in range(len(self._files)):
            if self._states[i].get():
                paths.append(self._folder + '\\' + self._files[i])
        return paths

    def _update_files(self):
        self._files = os.listdir(self._folder)
        self._update_file_frame()

    def _update_file_frame(self):
        for button in self._file_buttons:
            button.destroy()

        self._states = [tk.IntVar() for _ in self._files]
        for i in range(len(self._files)):
            cb = tk.Checkbutton(self._file_frame, variable=self._states[i], text=self._files[i])
            self._file_buttons.append(cb)
            cb.grid(row=i + 1, column=0, sticky="w")


class App(tk.Frame):

    def __init__(self, master, folders):
        super().__init__(master)

        self._folders = folders

        self._fig = plt.figure(figsize=(16, 8))

        self._set_folders_list()

    def _set_folders_list(self):
        self._folders_list = tk.Frame(self)
        for folder in self._folders:
            button = tk.Button(self._folders_list, text=folder, command=lambda f=folder: self._open_folder(f))
            button.pack(fill='x', padx=1, pady=1)
        self._folders_list.pack()

    def _open_folder(self, folder):
        self._folders_list.destroy()
        self._folder_frame = FolderFrame(self, folder, self._fig, self._back)
        self._folder_frame.pack()

    def _back(self):
        self._folder_frame.destroy()
        self._set_folders_list()


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
