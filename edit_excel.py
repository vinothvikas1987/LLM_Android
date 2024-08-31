
import os
import pandas as pd
from openpyxl import load_workbook

DEFAULT_GROUPS = [ 
    'Mobile', 'Broadband', 'TDS', 'Salary', "Mobile Payment", 
    'Biowaste', 'Investment and Deposits', 'Loan', 'Rent', 'EB', 
    'UPI Payment', 'OTT', 'Swiggy', 'Others'
]

# EXCEL_PATH = r'C:\Users\admin\Desktop\AUNTY\android.xlsx'.
EXCEL_PATH = r'./android.xlsx'



def extract_year(date):
    return pd.to_datetime(date).strftime('%Y')


def extract_month(date):
    return pd.to_datetime(date).strftime('%B')


def create_new_sheet(writer, year):
    df_new = pd.DataFrame({'Group': DEFAULT_GROUPS})
    for month in pd.date_range(start='1/1/2024', end='12/31/2024', freq='MS').strftime("%B"):
        df_new[month + '_Details'] = ''
        df_new[month + '_Date'] = ''
    df_new.to_excel(writer, sheet_name=year, index=False)


def save_to_excel(cumulative_results):
    df = pd.DataFrame(cumulative_results)
    df['Year'] = df['date'].apply(extract_year)
    df['Month'] = df['date'].apply(extract_month)
    df['Month_Details'] = df.apply(lambda x: f"{x['Month']}_Details", axis=1)
    df['Month_Date'] = df.apply(lambda x: f"{x['Month']}_Date", axis=1)
    df[df['Month_Details']] = df['details']
    df[df['Month_Date']] = df['date']

    if os.path.exists(EXCEL_PATH):
        with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            workbook = writer.book
            for year in df['Year'].unique():
                if year not in workbook.sheetnames:
                    create_new_sheet(writer, year)
            
            for year, year_group in df.groupby('Year'):
                df_excel = pd.read_excel(EXCEL_PATH, sheet_name=year)
                for month in pd.date_range(start='1/1/2024', end='12/31/2024', freq='MS').strftime("%B"):
                    df_excel[month + '_Details'] = df_excel[month + '_Details'].fillna('').astype(str)
                    df_excel[month + '_Date'] = df_excel[month + '_Date'].fillna('').astype(str)

                for i, row in year_group.iterrows():
                    row['date'] = pd.to_datetime(row['date'])
                    month = row['date'].strftime('%B')
                    col_details = f"{month}_Details"
                    col_date = f"{month}_Date"            
                    details = row[col_details]
                    date = row[col_date]

                    matching_idx = df_excel[df_excel['Group'] == row['Group']].index

                    if not matching_idx.empty:
                        idx = matching_idx[0]
                        new_row = {'Group': '', month + '_Details': details, month + '_Date': date}
                        df_excel = pd.concat([df_excel.iloc[:idx+1], pd.DataFrame([new_row]), df_excel.iloc[idx+1:]]).reset_index(drop=True)
                    else:
                        new_row = {'Group': row['Group'], month + '_Details': details, month + '_Date': date}
                        df_excel = pd.concat([df_excel, pd.DataFrame([new_row])], ignore_index=True)

                for group in DEFAULT_GROUPS:
                    if group not in df_excel['Group'].values:
                        new_row = {'Group': group}
                        for month in pd.date_range(start='1/1/2024', end='12/31/2024', freq='MS').strftime("%B"):
                            new_row[month + '_Details'] = ''
                            new_row[month + '_Date'] = ''
                        df_excel = pd.concat([df_excel, pd.DataFrame([new_row])], ignore_index=True)

                df_excel.to_excel(writer, sheet_name=year, index=False, header=True)
    else:
        with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl') as writer:
            for year, year_group in df.groupby('Year'):
                create_new_sheet(writer, year)
                df_excel = pd.DataFrame(columns=['Group'])

                for month in pd.date_range(start='1/1/2024', end='12/31/2024', freq='MS').strftime("%B"):
                    df_excel[month + '_Details'] = [''] * len(DEFAULT_GROUPS)
                    df_excel[month + '_Date'] = [''] * len(DEFAULT_GROUPS)

                for i, row in year_group.iterrows():
                    group = row['Group']
                    details = str(row['details'])
                    date = str(row['date'])
                    month = row['Month']
                    group_idx = df_excel[df_excel['Group'] == group].index

                    if not group_idx.empty:
                        idx = group_idx[0]
                        df_excel.at[idx, month + '_Details'] += f"{details}\n"
                        df_excel.at[idx, month + '_Date'] += f"{date}\n"
                    else:
                        new_row = {'Group': group, month + '_Details': details, month + '_Date': date}
                        df_excel = pd.concat([df_excel, pd.DataFrame([new_row])], ignore_index=True)

                for group in DEFAULT_GROUPS:
                    if group not in df_excel['Group'].values:
                        new_row = {'Group': group}
                        for month in pd.date_range(start='1/1/2024', end='12/31/2024', freq='MS').strftime("%B"):
                            new_row[month + '_Details'] = ''
                            new_row[month + '_Date'] = ''
                        df_excel = pd.concat([df_excel, pd.DataFrame([new_row])], ignore_index=True)

                df_excel.to_excel(writer, sheet_name=year, index=False, header=True)


# def flush_temp_storage(temp_storage):
#     success = True
#     for result in temp_storage:
#         try:
#             save_to_excel(result)
#         except Exception as e:
#             print(f"Failed to save to Excel: {e}")
#             success = False
#             break
#     if success:
#         temp_storage.clear()
