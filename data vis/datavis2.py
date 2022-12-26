# imports
import matplotlib.pyplot as plt
import pandas as pd

# read in csv
data_clean = pd.read_csv('data_clean.csv')
data_clean_restricted = data_clean.loc[data_clean['Age'] < 30]

def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{:.1f}%'.format(pct, v=val)
        return my_format

ratings_key = ['Poor', 'Fair', 'Adequate', 'Good', 'Exceptional']

# combine all plots
figure, axis = plt.subplots(6, 4)

colors = {'Poor': 'violet',
        'Fair': 'magenta',
        'Adequate': 'Orange',
        'Good': 'Red',
        'Exceptional': 'Cyan'}


data_ls=[[data_clean['Creative1'].to_list(), data_clean['Creative2'].to_list(), data_clean_restricted['Creative1'].to_list(), data_clean_restricted['Creative2'].to_list()],
[data_clean['Creative3'].to_list(), data_clean['Creative4'].to_list(), data_clean_restricted['Creative3'].to_list(), data_clean_restricted['Creative4'].to_list()],
[data_clean['Cohesive1'].to_list(), data_clean['Cohesive2'].to_list(), data_clean_restricted['Cohesive1'].to_list(), data_clean_restricted['Cohesive2'].to_list()],
[data_clean['Cohesive3'].to_list(), data_clean['Cohesive4'].to_list(), data_clean_restricted['Cohesive3'].to_list(), data_clean_restricted['Cohesive4'].to_list()],
[data_clean['Compelling1'].to_list(), data_clean['Compelling2'].to_list(), data_clean_restricted['Compelling1'].to_list(), data_clean_restricted['Compelling2'].to_list()],
[data_clean['Compelling3'].to_list(), data_clean['Compelling4'].to_list(), data_clean_restricted['Compelling3'].to_list(), data_clean_restricted['Compelling4'].to_list()]]

name_ls = [['Crea1', 'Crea2', 'CreaClip1', 'CreaClip2'],
['Crea3', 'Crea4', 'CreaClip3', 'CreaClip4'],
['Coh1', 'Coh2', 'CohClip1', 'CohClip2'],
['Coh3', 'Coh4', 'CohClip3', 'CohClip4'],
['Com1', 'Com2', 'ComClip1', 'ComClip2'],
['Com3', 'Com4', 'ComClip3', 'ComClip4']]

for j in range(len(data_ls)):
    ls = data_ls[j]
    list1 = ls[0]
    list2 = ls[1]
    list3 = ls[2]
    list4 = ls[3]

    name = name_ls[j]
    text1 = name[0]
    text2 =name[1]
    text3=name[2]
    text4=name[3]

    # get unique
    list1_unique = set(list1)
    list1_labels = []
    list1_count = []
    for uni in list1_unique:
        list1_count.append(list1.count(uni))
        list1_labels.append(ratings_key[uni])

    axis[j, 0].pie(list1_count, labels=list1_labels, autopct=autopct_format(list1), colors=[colors[key] for key in list1_labels])
    axis[j,0].set_title(text1+', '+f'{sum(list1)/(4*len(list1))*100:.3g}')

    # get unique
    list2_unique = set(list2)
    list2_labels = []
    list2_count = []
    for uni in list2_unique:
        list2_count.append(list2.count(uni))
        list2_labels.append(ratings_key[uni])


    axis[j, 1].pie(list2_count, labels=list2_labels, autopct=autopct_format(list2), colors=[colors[key] for key in list2_labels])
    axis[j,1].set_title(text2+', '+f'{sum(list2)/(4*len(list2))*100:.3g}')

    # get unique
    list3_unique = set(list3)
    list3_labels = []
    for label in list3_unique:
        list3_labels.append(ratings_key[label])
    list3_count = []
    for uni in list3_unique:
        list3_count.append(list3.count(uni))

    axis[j, 2].pie(list3_count, labels=list3_labels, autopct=autopct_format(list3), colors=[colors[key] for key in list3_labels])
    axis[j,2].set_title(text3+', '+f'{sum(list3)/(4*len(list3))*100:.3g}')

    # get unique
    list4_unique = set(list4)
    list4_labels = []
    for label in list4_unique:
        list4_labels.append(ratings_key[label])
    list4_count = []
    for uni in list4_unique:
        list4_count.append(list4.count(uni))

    axis[j,3].pie(list4_count, labels=list4_labels, autopct=autopct_format(list4), colors=[colors[key] for key in list4_labels])
    axis[j,3].set_title(text4+', '+f'{sum(list4)/(4*len(list4))*100:.3g}')

figure.suptitle('Results: N='+str(len(data_ls[0][0])) + ", N'="+str(len(data_ls[0][2])), fontsize=18)
plt.show()