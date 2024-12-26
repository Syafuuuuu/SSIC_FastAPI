# from datetime import datetime
# from collections import namedtuple
# from datetime import datetime

# #Festivities
# Festival = namedtuple('Festival', ['name','start_date','end_date', 'video', 'culture'])

# def read_festivals_from_file(file_path): 
#     festivals = [] 
#     with open(file_path, 'r') as file: 
#         for line in file: 
#             parts = line.strip().split(',')
            
#             if (len(parts) == 5):
#                 name, start_date, end_date, video, culture = parts
#                 festivals.append(Festival(name, start_date, end_date, video, culture))
#             else:
#                 name, start_date, end_date, video = parts
#                 festivals.append(Festival(name, start_date, end_date, video, None))                
            
#     print(festivals)
#     return festivals

    
# def date_checker(date_str, festivals):
#     input_date = datetime.strptime(date_str, "%Y-%m-%d")
    
#     for festival in festivals:
#         start_date = datetime.strptime(festival.start_date, "%Y-%m-%d")
#         end_date = datetime.strptime(festival.end_date, "%Y-%m-%d")
        
#         if start_date <= input_date <= end_date: 
#             return True, festival.video, festival.culture if festival.culture else ""
    
#     return False, "", ""



# date = "2025-02-09"
# Gate1Flag, Gate1Video, Gate1Culture = date_checker(date, read_festivals_from_file("./static/gen_fest.txt"))
# Gate2Flag, Gate2Video, Gate2Culture = date_checker(date, read_festivals_from_file("./static/cult_fest.txt"))
    
# if(Gate1Flag):
#     print("General Festival code")
#     print(f"The video: {Gate1Video}")
#     print(f"Part of culture: {Gate1Culture}")
# elif(Gate2Flag):
#     print("Cultural Festival code")
#     print(f"Video: {Gate2Video}")
#     print(f"Part of culture: {Gate2Culture}")
# else:
#     print("Date missed")


#---------------------------------------------------------------------------------------

def and_function(array1, array2):
    if len(array1) != len(array2):
        raise ValueError("Arrays must be of the same length")
    return [a and b for a, b in zip(array1, array2)]

def count_similar_interests(selection_array, willing_array, interest_array):
    # Initialize an array to store the count of similarities for each interest
    interest_count = [0] * len(interest_array[0])
    
    for selected_agent in selection_array:
        for willing_agent in willing_array:
            # Skip self-comparison
            if selected_agent == willing_agent:
                continue
            
            # Perform AND operation between the interest arrays of the selected and willing agents
            and_result = and_function(interest_array[selected_agent], interest_array[willing_agent])
            
            # Increment the count for each similar interest
            for index, value in enumerate(and_result):
                if value:
                    interest_count[index] += 1

    return interest_count

# Example usage
selection_array = [0, 1, 4]
willing_array = [0, 1, 3, 4]
interest_array = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Agent 0 interests
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Agent 1 interests
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 2 interests
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # Agent 3 interests
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   # Agent 4 interests
]

result = count_similar_interests(selection_array, willing_array, interest_array)
print(result)
