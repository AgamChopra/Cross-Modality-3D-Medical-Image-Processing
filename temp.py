import numpy as np
import pandas as pd


def get_usable_subjects(csv):
    '''
    Generate a dictionary of usable subjects based on some criteria.
    **Specific to ADNI3**
    Parameters
    ----------
    csv : pandas dataframe
        ADNI3 csv file related to downloaded data.
    Returns
    -------
    usable_subjects : dictionary
        usable subjects.
    usable : int
        number of usable subjects.
    '''
    subjects = np.array(csv['Subject'])
    return subjects


def generate_address_dict(matched_csv, mri_csv, pet_csv):#{'PID':[T1_folder,T2_folder,PET_folder],...}
    '''
    Generate dictionary of usable patient ids with T1, T2-FLAIR, and PET folders as in the ADNI3 csv file
    **Specific to ADNI3**
    Parameters
    ----------
    csv : pandas dataframe
        ADNI3 csv file related to downloaded data.
    Returns
    -------
    data : dictonary
        dictionary of modality file names by patient ids extracted from the ADNI3 csv file.
    '''
    usable_subjects = get_usable_subjects(matched_csv)    
    data = {}
    groups = []
    for ids in usable_subjects:
        val = {}      
        mri = mri_csv.loc[mri_csv['Subject'] == ids].to_numpy()    
        pet = pet_csv.loc[pet_csv['Subject'] == ids].to_numpy()    
        
        for b in mri:
            if b[6] == 'MRI':
                if 'Sag' in b[7]:
                    if 'Acc' in b[7] and 'ND' not in b[7]:
                        val['T1'] = b[7].replace(' ', '_')             
                    elif 'FLAIR' in b[7]:
                        val['T2'] = b[7].replace(' ', '_')   
                        
        for b in pet:
            if b[6] == 'PET':
                val['PET'] = b[7].replace(':', '_').replace(' ', '_').replace('/','_').replace('(','_').replace(')','_').replace('_Tau','') 
                group = b[2]
        
        try:
            data[ids] = [val['T1'],val['T2'],val['PET']]
            groups.append(group)
        except:
            continue
    
    print(groups, len(groups))
    ad = 0
    mci = 0
    cn = 0
    for p in groups:
        if p == 'AD':
            ad += 1
        elif p == 'MCI':
            mci += 1
        else:
            cn += 1
    print(ad,mci,cn)
    return data


def get_data_address_list(csv=[], mri_path = 'ad_project/data/final_adni/mri/ADNI/', pet_path = 'ad_project/data/final_adni/pet/ADNI/'):
    '''
    Generate list of usable relative data file paths
    **Specific to ADNI3**
    Parameters
    ----------
    csv : pandas dataframe
        ADNI3 csv file related to downloaded data.
    file_path : string, optional
        relative folder path of target folder. The default is 'ad_project/data/initial_only/intial_only/ADNI/'.
    Returns
    -------
    data_ : list of lists
        list of relative filepaths extracted from the ADNI3 csv file.
    '''
    data= generate_address_dict(csv[0],csv[1],csv[2])
    
    keys = data.keys()    
    data_ = []
    
    for pid in keys:
        temp = []  
        for i in range(len(data[pid])):
            if i < 2:
                temp.append(mri_path + pid + '/' + data[pid][i])
            else:
                temp.append(pet_path + pid + '/' + data[pid][i])
        data_.append(temp)
   
    return data_


def main_iterator(temp = '/home/agam/Documents/temp', out = 'temp/outF', 
                 root = '/home/agam/Desktop/', matched = '/home/agam/Desktop/ad_project/data/final_adni/matched.csv', 
                 pet = '/home/agam/Desktop/ad_project/data/final_adni/pet_tau.csv', mri = '/home/agam/Desktop/ad_project/data/final_adni/mri.csv'):
    
    #CSV LOGIC     
    csv = []
    csv.append(pd.read_csv(matched))
    csv.append(pd.read_csv(mri))
    csv.append(pd.read_csv(pet))
    adrs = get_data_address_list(csv)
    #print(adrs,len(adrs))
    

if __name__ == "__main__":
    main_iterator()