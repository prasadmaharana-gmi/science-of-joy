# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:08:43 2023

@author: Prasad Maharana
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import datetime
import pandas as pd
import time


def print_time():
    now = datetime.datetime.now()
    return now
    

def clear_text(Webelement):
     Webelement.send_keys(Keys.CONTROL + "a")
     Webelement.send_keys(Keys.DELETE)
     
     

def send_upc(upc_id):
    try:
        search_key=[]
        search_key.append(upc_id)
        search_key.append(upc_id[1:])
        final_key = ' '.join(str(el) for el in search_key)
        search_box = driver.find_element(By.ID, "applied-filters-typeahead")
        # enter your name in the search box
        search_box.send_keys(final_key)
        search_box.send_keys(Keys.RETURN)  # submit the search
        time.sleep(4)
        results = driver.find_elements(By.CSS_SELECTOR, "#root > div > div.Layout__Container-sc-12wjlu5-0.ksgQJp > div.Layout__Main-sc-12wjlu5-1.gUqgKv > div > div > div.Card-sc-1opdot8-0.styled__MainCard-eg58na-3.ccXGR > div.ProductGrid__Grid-gpbxew-0.fVviLt > a")
        
    except:
        results = []
    # if len(results) < 1 and len(upc_id) >= 11:  # retry with one less digit from the start
    #     results = send_upc(upc_id[1:])
    # time.sleep(1)
    clear_text(search_box)
    return results



def data_formatter(element):
    title=''
    upc_card = search[0].find_element(By.XPATH, '//*[@id="root"]/div/div[3]/div[2]/div/div/div[2]/div[1]/a/div[2]/div/h5').text
    title_card = search[0].find_element(By.XPATH,'//*[@id="root"]/div/div[3]/div[2]/div/div/div[2]/div[1]/a/div[2]')
    title_attr = title_card.find_elements(By.TAG_NAME,'h5')
    if len(title_attr)>1:
        for el in title_attr[:-2]:
            title+=str(el.text)+" "
        title=title.replace("oz","OUNCE")
    return upc_card,title        
            
    
    
# Read the input file
root_folder=r'C:\Users\G670813\OneDrive - General Mills\Desktop\upc_mapping\UPC_mapping'

input_file = pd.read_excel(root_folder+'/upc_list.xlsx',
                           sheet_name='Sheet1', dtype={"UPC": "string"})
#initialize browser
driver = webdriver.Chrome()
driver.get("https://app.labelinsight.com/explore-specs/search")
time.sleep(3)

# Login and submit
login = driver.find_element(By.ID, "userEmail").send_keys('prasad.maharana@genmills.com')
pwd = driver.find_element(By.ID, "userPassword").send_keys('Prasannamah123$')
submit = driver.find_element(By.XPATH, '//*[@id="root"]/div/div[3]/div/div/main/form/div[3]/button')
submit.click()
time.sleep(5)



#**********************************************************************
# make sure to debug this point for login authentication
#**********************************************************************
            
upc_count=len(input_file)            
output = pd.DataFrame(columns=['Connect_UPC', 'Connect_Desc','NIQ_LI_UPC', 'NIQ_desc'])

#print start time 
start_time=print_time()
file_name="/Mapped UPCs_"+str(start_time)[:-16]+".xlsx"


#print initial stats
print("Total UPCs Found: {}".format(upc_count))
print("Start Time of Mapping: {}".format(start_time.time()))


for index, row in input_file.iterrows():
    search = send_upc(row['UPC'])
    if len(search) < 1:
        new_upc,title = "NA","NA"

    elif len(search) == 1:
        new_upc,title = data_formatter(search)
    else:
        ("Multiple UPCs found. Conflict.")
        new_upc,title= "Conflict","Conflict"
    output = output.append({'Connect_UPC': row['UPC'], 'Connect_Desc': row['ITEM'],'NIQ_LI_UPC': new_upc,'NIQ_desc':title}, ignore_index=True)
    if len(output)%100==0:
        print("Completed {} rows. Out of {}".format(len(output),upc_count))
output.to_excel(root_folder+file_name)

end_time=print_time()
print("End Time of Mapping: {}".format(end_time.time()))
delta=end_time-start_time
print("Total Time Elapsed: {}".format(delta.total_seconds()))
print("Average time per UPC scan: {}".format(delta.total_seconds()/upc_count))
driver.quit()
