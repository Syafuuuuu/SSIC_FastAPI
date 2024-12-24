from datetime import datetime
from collections import namedtuple
from datetime import datetime

#Festivities
Festival = namedtuple('Festival', ['name','start_date','end_date', 'video', 'culture'])

def read_festivals_from_file(file_path): 
    festivals = [] 
    with open(file_path, 'r') as file: 
        for line in file: 
            parts = line.strip().split(',')
            
            if (len(parts) == 5):
                name, start_date, end_date, video, culture = parts
                festivals.append(Festival(name, start_date, end_date, video, culture))
            else:
                name, start_date, end_date, video = parts
                festivals.append(Festival(name, start_date, end_date, video, None))                
            
    print(festivals)
    return festivals

    
def date_checker(date_str, festivals):
    input_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    for festival in festivals:
        start_date = datetime.strptime(festival.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(festival.end_date, "%Y-%m-%d")
        
        if start_date <= input_date <= end_date: 
            return True, festival.video, festival.culture if festival.culture else ""
    
    return False, "", ""



date = "2025-02-09"
Gate1Flag, Gate1Video, Gate1Culture = date_checker(date, read_festivals_from_file("./static/gen_fest.txt"))
Gate2Flag, Gate2Video, Gate2Culture = date_checker(date, read_festivals_from_file("./static/cult_fest.txt"))
    
if(Gate1Flag):
    print("General Festival code")
    print(f"The video: {Gate1Video}")
    print(f"Part of culture: {Gate1Culture}")
elif(Gate2Flag):
    print("Cultural Festival code")
    print(f"Video: {Gate2Video}")
    print(f"Part of culture: {Gate2Culture}")
else:
    print("Date missed")
