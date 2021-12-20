from typing import Counter
import pandas as pd
import matplotlib.pyplot as plt
import collections

Data_WuzzufJobs = pd.read_csv("Wuzzuf_Jobs.csv")


Data_WuzzufJobs.describe()      # display structure and summary of the data.

# clean the data (null, duplications)
Data_WuzzufJobs.drop_duplicates(
    subset=["Title", "Company"], keep="first", inplace=True)

# ---------

# 4.count the jobs for each company
# 5.show step 4 in a pie chart
x = Data_WuzzufJobs['Company'].value_counts()
print(x[0:5])
e1 = x.values[:5]
mylabels = x.index[:5]
plt.pie(e1, labels=mylabels)
plt.show()

# ----------

# 6.the most popular job titles
job_titles = Data_WuzzufJobs['Title'].value_counts()
print(job_titles[0:5])
# 7.the most popular job titles in bar chart
T1 = job_titles.index[0:5]
T2 = job_titles.values[0:5]
plt.bar(T1, T2, color='red', width=0.3)
plt.ylabel('value')
plt.xlabel('Job Title')
plt.title('the most popular job titles')
plt.show()

# ---------

# 8.Find out the most popular areas?
areas = Data_WuzzufJobs['Location'].value_counts()
print(areas[0:5])
# 9.Find out the most popular areas? in bar chart
L2 = areas.values[:5]
L1 = areas.index[:5]
plt.bar(L1, L2, color='red', width=0.3)
plt.xlabel("City")
plt.ylabel("value")
plt.title("the most popular areas?")
plt.show()


# --------

# 10.Print skills one by one
_skill = dict()
for skills in Data_WuzzufJobs['Skills']:
    for skill in skills.split(','):
        if skill in _skill.keys():
            _skill[skill] += 1
        else:
            _skill[skill] = 1
collections_skills = Counter(_skill)
common_skills = dict(collections_skills.most_common(5))
print("The Most Important Skills Required ? ")
i = 0
for skill, count in common_skills.items():
    i += 1
    print(i, "%s : %d" % (skill, count))
