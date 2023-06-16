from matplotlib import rc
# getting necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

n = 50
m = 2000
data = []
mean = np.zeros(n)
for i in range(50):
    data.append(np.load(str(i) + ".npy"))
data = np.vstack(data)[0:n + 1, 0:m]

for j in range(n):
    mean[j] = np.mean(data[j, :])

print(data.shape)
# print(data)

epochInd = [i for i in range(50)]
toDel = [i for i in range(0, 50, 2)]
print(len(toDel))
dataList = []
for i in range(n):
    if i not in toDel:
        tmp = np.zeros((int(m), 3))
        tmp[:, 1] = np.ones(int(m)) * epochInd[i]
        tmp[:, 2] = np.ones(int(m)) * mean[i]
        for j in range(int(m)):
            tmp[j, 0] = data[i, j]
        dataList.append(tmp)

data = np.vstack(dataList)
vals = data[:, 0].tolist()
labels = data[:, 1].astype('int').astype('str').tolist()
means = data[:, 2].tolist()
# print(means)
# print(labels)


# Col = ["Vals", "Epoch"]
# df = pd.DataFrame(data, columns = Col)
df = pd.DataFrame(data={'Epoch': labels, 'Vals': vals, 'Epoch_Mean': means})
print(df)
# 
# 
# sns.kdeplot(data=df, x="Vals", hue='Epoch')
# plt.savefig("figure.pdf", dpi=300, bbox_inches='tight')


# #
# # # getting the data
# temp = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2016-weather-data-seattle.csv') # we retrieve the data from plotly's GitHub repository
# temp['month'] = pd.to_datetime(temp['Date']).dt.month # we store the month in a separate column
# 
# # we define a dictionnary with months that we'll use later
# month_dict = {1: 'january',
#               2: 'february',
#               3: 'march',
#               4: 'april',
#               5: 'may',
#               6: 'june',
#               7: 'july',
#               8: 'august',
#               9: 'september',
#               10: 'october',
#               11: 'november',
#               12: 'december'}
# 
# # we create a 'month' column
# temp['month'] = temp['month'].map(month_dict)
# # # 
# print(temp)
# # we generate a pd.Serie with the mean temperature for each month (used later for colors in the FacetGrid plot), and we create a new column in temp dataframe
# month_mean_serie = temp.groupby('month')['Mean_TemperatureC'].mean()
# temp['mean_month'] = temp['month'].map(month_mean_serie)
# 
# 
# we generate a color palette with Seaborn.color_palette()
pal = sns.color_palette(palette='coolwarm', n_colors=20)

# # in the sns.FacetGrid class, the 'hue' argument is the one that is the one that will be represented by colors with 'palette'
g = sns.FacetGrid(df, row='Epoch', hue='Epoch_Mean', aspect=15, height=0.80, palette=pal)
# 
# # then we add the densities kdeplots for each month
g.map(sns.kdeplot, 'Vals',
      bw_adjust=1, clip_on=False, warn_singular=False,
      fill=True, alpha=1, linewidth=1.5)

# # # # here we add a white line that represents the contour of each kdeplot
# g.map(sns.kdeplot, 'Vals', 
#       bw_adjust=1, clip_on=False, warn_singular=False,
#       color="w", lw=2)
# # 
# # here we add a horizontal line for each plot
g.map(plt.axhline, y=0,
      lw=2, clip_on=False)
# # 
# we loop over the FacetGrid figure axes (g.axes.flat) and add the month as text with the right color
# notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
for i, ax in enumerate(g.axes.flat):
    ax.text(33, 0.03, i,
            fontweight='bold', fontsize=15,
            color=ax.lines[-1].get_color())

# we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
g.fig.subplots_adjust(hspace=-0.88)

# eventually we remove axes titles, yticks and spines
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)

plt.setp(ax.get_xticklabels(), fontsize=15, fontweight='bold')
plt.xlabel("ACAA1", fontweight='bold', fontsize=15)
g.fig.suptitle('tmp',
               ha='right',
               fontsize=20,
               fontweight=20)
# plt.xlim(xmin=0)
# plt.show()
plt.savefig("figure.pdf", dpi=300, bbox_inches='tight')
