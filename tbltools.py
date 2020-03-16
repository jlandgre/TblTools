import pandas as pd
import numpy as np

#Create summary DataFrame of a measure column, fld_meas, by list of category columns, lst_summby
#Returns the summary DataFrame sorted by the measure's values in descending order; index in ascending order
#Version 1/6/2020
def field_summary(df, lst_summby, fld_meas, fld_name_ct, fld_name_total):
    keep_cols = lst_summby.copy()
    keep_cols.append(fld_meas)
    df_summ = df[keep_cols].groupby(lst_summby).agg({lst_summby[0]:'count',fld_meas:'sum'})

    #Rename the columns as specified in arguments
    df_summ.columns = [fld_name_ct,fld_name_total]

    #create integer index; make fld_summby into table column
    df_summ.reset_index(inplace=True)

    return df_summ

#Add a ranking column to a DataFrame based on a numeric column, fld_name_total. If lst_subcats
#is not specified, ranking will be by overall table sort; otherwise by list of column name subcategories
#Version 1/7/2020
def add_ranking(df, fld_name_ranking, fld_name_total, lst_subcats=None):

    if lst_subcats is None: lst_subcats = []
    lst_sortby = []
    lst_ascend = []

    #Replace the ranking column if pre-existing
    if fld_name_ranking in df.columns: df.drop(columns=[fld_name_ranking],inplace=True)

    #Build lists for sortby order and ascending/descending (for categories, True --> alphabetical)
    lst_sortby = lst_sortby + lst_subcats
    for s in lst_subcats:
        lst_ascend.append(True)
    lst_sortby.append(fld_name_total)
    lst_ascend.append(False)

    #Sort the table and insert ranking column
    df.sort_values(by=lst_sortby,inplace=True,ascending=lst_ascend)
    df.reset_index(drop=True,inplace=True)
    df.insert(loc=len(df.columns), column=fld_name_ranking, value=0)

    #Initialize sub-category, current values
    lst_curvals = []
    for subcat in lst_subcats:
        lst_curvals.append(df.loc[0,subcat])

    #Populate the ranking
    cur_rank = 0
    for row in range(0,len(df.index)):

        #Default to increment the rank
        cur_rank += 1

        #Test whether subcategory values (if any) have changed
        for i in range(len(lst_subcats)):
            if df.loc[row,lst_subcats[i]] != lst_curvals[i]:
                cur_rank = 1
                for j, subcat in enumerate(lst_subcats):
                    lst_curvals[j] = df.loc[row,subcat]
                break

        #Write the ranking to the DataFrame row
        df.loc[row,fld_name_ranking] = cur_rank

    df[fld_name_ranking].astype('int32')
    return df

#Create summary DataFrame to track numeric stats, null counts etc.
#Version of 1/15/2020
def table_summary_df(df):
    summary_df = pd.DataFrame(columns=['col_name','n_null','n_non-null',
                                       'n_unique','min','max','mean', 'dtype'])
    for i in range(0,len(df.columns)):
        col = df.columns[i]

        #Use .loc to populate the value AND add the row to DataFrame
        summary_df.loc[i,'col_name'] = col

        #populate statistics about each column
        summary_df['n_null'].at[i] = len(df.index) - df[col].count()
        summary_df['n_non-null'].at[i] = df[col].count()
        if df[col].dtype == 'object':
            summary_df['n_unique'].at[i] = df[col].nunique()
        elif df[col].dtype == 'int64' or 'float64':
            summary_df['min'].at[i] = df[col].min()
            summary_df['max'].at[i] = df[col].max()
            summary_df['mean'].at[i] = df[col].mean()
            if df[col].dtype == 'datetime64[ns]':
                summary_df['mean'].at[i] = np.nan
        summary_df['dtype'].at[i] = df[col].dtype
    return(summary_df)

#Returns consolidated df with coincident rows rolled up. Coincident refers to rows
#having same date/time where rows are populated in various columns.  In case of rows where same
#column is populated as previous, coincident row, no consolidation occurs unless
#column is in a separate, 'override' list.
#Version of 3/16/20 - uses .loc instead of .iloc; consolidates "downward"

def RollupCoincidentRows(df_in, dt_col, lst_cols, lst_override, IsFlagConflicts, IsDeleteCoinc):

    df = df_in.copy()

    #Add flag columns and populate with defaults
    kp_col, confl_col, coinc_col = 'keep', 'RowConflict', 'IsCoincident'
    df[kp_col], df[confl_col], df[coinc_col] = True, False, False

    #Start with second row
    idxFirst = df.index.values[0]
    for idx, row in df.iloc[1:].iterrows():

        #Skip rows already flagged for deletion
        idxPrev = IndexPrev(df, idx)
        while idxPrev == idxFirst and not df[kp_col].loc[idxPrev]:
            idxPrev = IndexPrev(df, idxPrev)

        #Consolidate if idx and idxPrev are coincident and idx's data don't conflict
        if row[dt_col] == df[dt_col].loc[idxPrev]:
            df[coinc_col].loc[idx], df[coinc_col].loc[idxPrev] = True, True

            #Default is no conflicts; keep=False for row i
            IsIrresolvable, IsConflict = False, False
            df[kp_col].loc[idxPrev] = False

            #Check each column
            for col in lst_cols:
                if not IsRowConflict(df, idx,idxPrev, col):
                    if IsNullCell(df, idx, col): df[col].loc[idx] = df[col].loc[idxPrev]

                elif col in lst_override:
                    IsConflict = True
                else: IsConflict, IsIrresolvable = True, True

                #Flag conflict whether overridden or not
                if IsConflict and IsFlagConflicts:
                    df[confl_col].loc[idxPrev], df[confl_col].loc[idx] = True, True

            #Don't drop the row if unresolved conflicts
            if IsIrresolvable: df[kp_col].loc[idxPrev] = True


    #Return after dropping flagged rows and Boolean columns
    if IsDeleteCoinc:
        if not IsFlagConflicts: df.drop(confl_col, axis=1, inplace=True)
        df.drop(coinc_col, axis=1, inplace=True)
        return df[df[kp_col]].drop(kp_col, axis=1)
    else:
        return df

def IsRowConflict(df, idx, idxPrev, col):
    if not IsNullCell(df, idxPrev, col):
        if not IsNullCell(df, idx, col): return True
    return False

#TRUE if row i of df col is NaN
def IsNullCell(df, idx, col):
    if pd.isnull(df[col].loc[idx]): return True
    return False

#Returns the index of the previous row
def IndexPrev(df, idx):
    return df.index.values[df.index.get_loc(idx) - 1]
