import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


def read_csv_file():
    print('Read csv file...')
    root_path = r'/data_final.csv'
    data = pd.read_csv(root_path, encoding="gbk")
    # print(data.describe())
    # print(data.head())
    # print(data.info())

    return data


def display_time_intervals():
    print('Display the time interval between two physical examinations...')
    data = read_csv_file()
    examination_DateTime = data['examination_DateTime'].to_numpy()

    f_year_date = examination_DateTime[0::3]
    s_year_date = examination_DateTime[1::3]
    t_year_date = examination_DateTime[2::3]

    f2s_interval = []
    s2t_interval = []
    for i in range(0, len(f_year_date)):
        f_date = datetime.strptime(str(f_year_date[i]), "%Y%m%d")
        s_date = datetime.strptime(str(s_year_date[i]), "%Y%m%d")
        t_date = datetime.strptime(str(t_year_date[i]), "%Y%m%d")

        f2s_days = (s_date - f_date).days
        s2t_days = (t_date - s_date).days

        f2s_interval.append(f2s_days)
        s2t_interval.append(s2t_days)

    f2s = np.array(f2s_interval)
    s2t = np.array(s2t_interval)

    print(f'Time interval between first and second physical examination, median: {np.median(f2s)} days, 25%: {np.percentile(f2s, 25)} days, 75%: {np.percentile(f2s, 75)} ')
    print(f'Time interval between second and third physical examination, median: {np.median(s2t)} days, 25%: {np.percentile(s2t, 25)} days, 75%: {np.percentile(s2t, 75)} ')

    # 设置全局字体大小
    plt.rcParams.update({'font.size': 16, 'font.family': 'Times New Roman'})

    bin_width = 30  # 每个组的宽度
    plt.figure(figsize=(8, 6))
    plt.hist(f2s, bins=range(min(f2s), max(f2s) + bin_width, bin_width), edgecolor='#FFF2CC', color='#DEEBF7')
    plt.xlabel('Days')
    plt.ylabel('Frequency')
    plt.title('Time interval between first and second physical examination', fontsize=18)

    plt.figure(figsize=(8, 6))
    plt.hist(s2t, bins=range(min(s2t), max(s2t) + bin_width, bin_width), edgecolor='#FFF2CC', color='#DEEBF7')
    plt.xlabel('Days')
    plt.ylabel('Frequency')
    plt.title('Time interval between second and third physical examination', fontsize=18)

    plt.show()


def del_variables_less_than_33():
    print('Remove those indicators with more than 67% missing values...')
    print('Determination of labels for dyslipidemia based on the "2023 China guidelines for lipid management"...')

    data = read_csv_file()

    df_new = pd.DataFrame()
    df_new['subject_ID'] = data['subject_ID']

    sex = data['sex']
    length = len(sex)

    # 指定要删除的列名称列表
    columns_to_remove = ['subject_ID', 'examination_DateTime']
    data = data.drop(columns=columns_to_remove)
    keys = data.keys()
    for key in keys:
        curr = data[key]
        if curr.isnull().sum()<(length/3):
            curr = [float(i) for i in curr]
            df_new[key] = curr

    TC = data['total cholesterol'].to_numpy()
    TG = data['triglycerides'].to_numpy()
    HDL = data['high-density lipoprotein cholesterol'].to_numpy()
    LDL = data['low-density lipoprotein cholesterol'].to_numpy()

    label = []
    for ii in range(length):
        if TC[ii]>=5.2 or TG[ii]>=1.7 or LDL[ii]>=3.4 or HDL[ii]<1.0:
            label.append(1)
        else:
            label.append(0)

    label = label[2::3]

    return df_new, label


def del_third_exam():
    # print('Delete the third physical examination record...')

    data, label = del_variables_less_than_33()
    # print(data.keys())
    data['Row_Num'] = data.groupby('subject_ID').cumcount()
    # 根据新列进行筛选，删除每组的最后一个数据
    data_new = data[data['Row_Num'] < data.groupby('subject_ID')['subject_ID'].transform('size') - 1]

    # 删除新添加的行号列
    data_new.drop('Row_Num', axis=1, inplace=True)

    return data_new, label


def interpolation_data():
    # print('Interpolate missing data...')

    data, label = del_third_exam()

    keys = ['sex', 'age', 'height', 'weight', 'systolic blood pressure',            # 0-4
       'diastolic blood pressure', 'pulse', 'body mass index', 'total protein',     # 5-8
       'albumin', 'globulin', 'gamma-glutamyl transpeptidase',                      # 9-11
       'alkaline phosphatase', 'total bilirubin', 'direct bilirubin',               # 12-14
       'indirect bilirubin', 'alanine aminotransferase',                            # 15-16
       'aspartate aminotransferase', 'albumin/globulin ratio',                      # 17-18
       'aspartate aminotransferase/ alanine aminotransferase',                      # 19
       'total cholesterol', 'triglycerides',                                        # 20-21
       'high-density lipoprotein cholesterol',                                      # 22
       'low-density lipoprotein cholesterol', 'glycated hemoglobin',                # 23-24
       'absolute value of lymphocytes', 'absolute neutrophil count',                # 25-26
       'platelet specific volume', 'monocyte ratio', 'eosinophil ratio',            # 27-29
       'basophil ratio', 'monocyte absolute value', 'large platelet ratio',         # 30-32
       'mean corpuscular hemoglobin',                                               # 33
       'red blood cell volume distribution width standard deviation',               # 34
       'hemoglobin', 'corpuscular specific volume', 'mean corpuscular volume',      # 35-37
       'mean corpuscular hemoglobin concentration',                                 # 38
       'coefficient of variation of red blood cell distribution width',             # 39
       'blood platelet count', 'mean platelet volume',                              # 40-41
       'platelet distribution width', 'absolute basophil count',                    # 42-43
       'absolute eosinophil count', 'neutrophil percentage',                        # 44-45
       'lymphocyte percentage', 'red blood cell count',                             # 46-47
       'white blood cell count', 'uric acid', 'burea nitrogen', 'creatinine',       # 48-51
       'thyrotropin', 'transparency', 'urinary bilirubin', 'urine ketone body',     # 52-55
       'urinary leukocyte esterase', 'urinary occult blood', 'color',               # 56-58
       'specific gravity', 'pH', 'urine nitrite', 'urinary protein',                # 59-62
       'urine glucose', 'urobilinogen', 'alpha-fetoprotein',                        # 63-65
       'carcinoembryonic antigen', 'glucose']                                       # 66-67

    nums_keys = ['age', 'height', 'weight', 'systolic blood pressure',
       'diastolic blood pressure', 'pulse', 'body mass index', 'total protein',
       'albumin', 'globulin', 'gamma-glutamyl transpeptidase',
       'alkaline phosphatase', 'total bilirubin', 'direct bilirubin',
       'indirect bilirubin', 'alanine aminotransferase',
       'aspartate aminotransferase', 'albumin/globulin ratio',
       'aspartate aminotransferase/ alanine aminotransferase',
       'total cholesterol', 'triglycerides',
       'high-density lipoprotein cholesterol',
       'low-density lipoprotein cholesterol', 'glycated hemoglobin',
       'absolute value of lymphocytes', 'absolute neutrophil count',
       'platelet specific volume', 'monocyte ratio', 'eosinophil ratio',
       'basophil ratio', 'monocyte absolute value', 'large platelet ratio',
       'mean corpuscular hemoglobin',
       'red blood cell volume distribution width standard deviation',
       'hemoglobin', 'corpuscular specific volume', 'mean corpuscular volume',
       'mean corpuscular hemoglobin concentration',
       'coefficient of variation of red blood cell distribution width',
       'blood platelet count', 'mean platelet volume',
       'platelet distribution width', 'absolute basophil count',
       'absolute eosinophil count', 'neutrophil percentage',
       'lymphocyte percentage', 'red blood cell count',
       'white blood cell count', 'uric acid', 'burea nitrogen', 'creatinine',
       'thyrotropin',
       'specific gravity', 'pH', 'alpha-fetoprotein',
       'carcinoembryonic antigen', 'glucose']

    category_keys = [
        'sex','transparency', 'urinary bilirubin', 'urine ketone body',
       'urinary leukocyte esterase', 'urinary occult blood', 'color',
        'urine nitrite', 'urinary protein', 'urine glucose', 'urobilinogen',
    ]

    # 优先按照分组进行插值
    for key in keys:
        if data[key].isnull().sum() != 0:
            data[key] = data.groupby('subject_ID')[key].ffill()
            data[key] = data.groupby('subject_ID')[key].bfill()

    # 对数值类型数据进行样条插值
    for key in nums_keys:
        if data[key].isnull().sum() != 0:
            data[key] = data[key].interpolate(method='spline', order=3)

    # 对类别类型数据进行众数插值
    for key in category_keys:
        if data[key].isnull().sum() != 0:
            mode_value = data[key].mode().iloc[0]
            data[key] = data[key].fillna(mode_value)

    # 删除subject_ID这一列
    data.drop('subject_ID', axis=1, inplace=True)

    return data, label


if '__main__' == __name__:
    display_time_intervals()
