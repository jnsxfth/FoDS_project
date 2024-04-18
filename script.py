import pandas as pd
import seaborn
from matplotlib import pyplot as plt


data_maths = pd.read_csv('./data/Maths.csv')
data_port = pd.read_csv('./data/Portuguese.csv')

"""*Explaining Columns*

school = (GP or MS)
sex = (F or M)
age = (15-22)
address = urban or rural (U or R)
famsize = 'less or equal to 3' or 'greater than 3' (LE3 or GT3)
Psatus = parent's cohabitation status: together or apart (T or A)
Medu = mother's education: none, primary, 5th to 9th grade, secondary, higher (0-4)
Fedu = father's education: none, primary, 5th to 9th grade, secondary, higher (0-4)
Mjob = mother's job: (teacher, health, services, at_home, other)
Fjob = father's job: (teacher, health, services, at_home, other)
reason = reason to choose school: close to home, reputation, course preverence, other (home, reputation, course, other)
guardian = student's guardian (mother, father, other)
traveltime = home to school travel time (1=1-<15min, 2=15-30min, 3=30-60min, 4=>1h)
studytime = weekly study time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)
failures = number of past class failures (n if 1<=n<3, else 4)
schoolsup = extra educational support (yes or no)
famsup = family educational support (yes or no)
paid = extra paid classes within the course subject (yes or no)
activities = extra-curricular activities (yes or no)
nursery = attended nursery school (yes or no)
higher = wants to take higher education (yes or no)
internet = internet access at home (yes or no)
romantic = with a romantic relationship (yes or no)
famrel = quality of family relationships (1=very bad to 5=excellent)
freetime = free time after school (1= very low to 5=very high)
goout = going out with friends (1=very low to 5=very high)
Dalc = workday alcohol consumption (1=very low to 5=very high)
Walc = weekend alcohol consumption (1=very low to 5=very high)
health = current health status (1=very bad to 5=very good)
absences = number of school absences (0-93)
G1-G3 = first/second/final grade (0-20)
"""
cat_cols = ['school', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
            'G1', 'G2', 'G3']
for data in [data_maths, data_port]:
    for col in cat_cols:
        data[col] = pd.Categorical(data[col])

def rename_education(data, parent):
    data[parent] = data[parent].cat.rename_categories(
        {0: "none", 1: "primary education", 2: "5th to 9th", 3: "secondary education", 4: "higher education"})


rename_education(data_maths, 'Fedu')
rename_education(data_port, 'Fedu')
rename_education(data_maths, 'Medu')
rename_education(data_port, 'Medu')

