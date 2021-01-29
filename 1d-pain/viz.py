import matplotlib.pyplot as plt


def plot_time_series(time, values, label):
    plt.figure(figsize=(10,6))
    plt.plot(time, values)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.title(label, fontsize=20)
    plt.grid(True)


def plot_many_time_series(time, values, title, labels):
    plt.figure(figsize=(10,6))
    for ind, v in enumerate(values):
        plt.plot(time, v, label='{}'.format(labels[ind]))
    plt.xlabel("t (s)", fontsize=20)
    plt.ylabel("Pain expression", fontsize=20)
    plt.title(title, fontsize=30)
    plt.legend(fontsize=20)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 5.5)
    fig.savefig('test2png.png', dpi=100)
    # plt.rcParams["figure.figsize"] = (30,2)
    plt.grid(True)
    
    plt.savefig('sparsepain.png')
