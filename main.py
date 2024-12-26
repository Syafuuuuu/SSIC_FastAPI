from collections import defaultdict
import random
from fastapi import FastAPI, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
import matplotlib.pyplot as plt
import io
import base64
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List
import numpy as np
from pathlib import Path
from fastapi.responses import StreamingResponse
from io import BytesIO
from datetime import datetime
from collections import namedtuple

# FastAPI app instance
app = FastAPI()

# Database setup
DATABASE_URL = "sqlite:///./agents.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Static files directory
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Add a global array to store TV coordinates
tv_positions = []
agents_list = []

GNumStep = 10000

#Translate iput for religion and culture (either or kinda thing)
def translate(input):
    match input:
        #   [0] - Malay/Islam
        #   [1] - Chinese/Christian
        #   [2] - Indian/Buddhism
        #   [3] - Others/Hinduism
        case 1:
            return [1,0,0,0]
        case 2:
            return [0,1,0,0]
        case 3:
            return [0,0,1,0]
        case 4:
            return [0,0,0,1]

def Gate2(date_str, festivals):
    input_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    #Check whole cultural festival dates
    for festival in festivals:
        start_date = datetime.strptime(festival.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(festival.end_date, "%Y-%m-%d")
        
        #If it falls on a cultural festival
        if start_date <= input_date <= end_date: 
            #Check if that cultural festival has any ready to interact members in this cluster
            
                #Does the 
            return True, festival.video, festival.culture if festival.culture else ""
    
    return False, "", ""
  
#Check for 3 things    
def Gate1(agent_behaviour, agent_int, agent_cult, int_count, int_avg, cult_count, cult_avg):
    
    flag = False
    content = ""
    
    #Check if there is 2 or more agents willing to interact and check
    # if((has_extrovert(agent_behaviour=agent_behaviour,type='willing') >= 2) or (has_extrovert(agent_behaviour=agent_behaviour,type='ready') >= 1)):
        # return flag, content
        
    
    #Check if there are extroverts (Ready to Interact) are present
    #   If got, check if those extroverts are interested in the 
    
    #Check if they speak the same language (culture)
    #   If they dont, default to eng/bm
    #   If they do, change content to that language
    
    # Ri_index, Li_index = extrovert_index(agent_behaviour=agent_behaviour)
    # if 
    
    
    
    return False


#Decision Model
def decision_model(date, agent_behaviour, interest_count, agent_interest, agent_culture, culture_count, interest_arr, culture_arr):
    
    print("")
    print("")
    print("")
    print("----------- | Decision Model Input | -----------")
    print(f"Date: {date}")
    print(f"Agent Behaviour: {agent_behaviour}")
    print(f"Interest Array: {agent_interest}")
    print(f"Interest Count: {interest_count}")
    print(f"Interest OR Values: {interest_arr}")
    print(f"Culture Array: {agent_culture}")
    print(f"Culture Count: {culture_count}")
    print(f"Culture OR Values: {culture_arr}")
    print("----------- | Decision Model Input | -----------")
    print("")
    print("")
    print("")
    
    #Festivities
    Festival = namedtuple('Festival', ['name','start_date','end_date', 'video', 'culture'])
    
    #Interest Video
    IntVid = namedtuple('IntVid', ['interest','lang','vidURL'])
    
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
                
        # print(festivals)
        return festivals
        
    def festival_checker(date_str, festivals):
        input_date = datetime.strptime(date_str, "%Y-%m-%d")
        
        #Check whole general festival dates
        for festival in festivals:
            start_date = datetime.strptime(festival.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(festival.end_date, "%Y-%m-%d")
            
            #if it falls on a specific festival
            if start_date <= input_date <= end_date: 
                
                #return details of that festival
                return True, festival.culture, festival.video
        
        return False, "", ""

    def read_interest_video_from_file(file_path): 
        
        #Interest Video
        IntVid = namedtuple('IntVid', ['interest','lang','vidURL'])
        
        interests_vids = [] 
        with open(file_path, 'r') as file: 
            for line in file: 
                parts = line.strip().split(',')
                
                interest, lang, vidURL = parts
                interests_vids.append(IntVid(interest, lang, vidURL))
             
                
        return interests_vids

    def interest_vid_decider(interest_index, lang_index, video_arr):
        for interest_video in video_arr:
            
            if((int(interest_index)==int(interest_video.interest)) and (int(lang_index)==int(interest_video.lang))):
                return interest_video.vidURL
            
        return "No such video"
        

    def AND_func(array1, array2):
        return [a and b for a, b in zip(array1, array2)]
    
    def highest_count(interest_count):
        # Find the maximum count
        max_count = max(interest_count)
        # Find all indices with the maximum count
        highest_indices = [index for index, count in enumerate(interest_count) if count == max_count]
        return highest_indices
    
    def has_extrovert(agent_behaviour, type):
        print("Enter has_extrovert()")
        count = 0
        #Set type of check
        if(type=="ready"):  #   Ri = agent[0]
            pos = 0
        if(type=="willing"):#   Li = agent[1]
            pos = 1
               
        print(f"Passed type check: {pos}")
                
        for agent in agent_behaviour:
            #Cout how many willing/ready extroverts
            print(f"Agent[pos] = {agent[pos]}")
            if(agent[pos]>=0.5):
                count = count + 1
        
        return count

    def extrovert_index(agent_behaviour):
        ready_to_interact_index = []
        willing_to_interact_index = []
        for index, agent in enumerate(agent_behaviour):
            #   Ri = agent[0]
            #   Li = agent[1]
            if(agent[0]>=0.5):
                ready_to_interact_index.append(index)
            if(agent[1]>=0.5):
                willing_to_interact_index.append(index)
                
        return ready_to_interact_index, willing_to_interact_index
    
    def has_common(ready_index, willing_index, subject_arr, start=None, end=None):
        
        flag = False
        index_of_highest_subj = -1
        
        print("")
        print("Debug")
        print(f"Subject: {subject_arr}")
        print(f"Example of subj: {subject_arr[0]}")
        print(f"Start: {start}")
        print(f"End: {end}")
        print(f"Processed Array: {subject_arr[0][start:end]}")
        
        for i in ready_index:
            for j in willing_index:
                
                if(i==j):
                    continue
                if(start==None and end==None):
                    and_result = AND_func(subject_arr[i], subject_arr[j])
                else:
                    and_result = AND_func(subject_arr[i][start:end], subject_arr[j][start:end])
                
                # Increment the count for each similar interest
                for index, value in enumerate(and_result):
                    if value:
                        interest_count[index] += 1 
                        flag = True            
        
        index_of_highest_subj = highest_count(interest_count)
        
        return flag, index_of_highest_subj
    
    def check_cult_fest_match(rel_arr, fest_index):
        print(f"Rel Arr: {rel_arr}")
        for rel in rel_arr:
            if(int(rel)==int(fest_index)):
                return True
            
        return False
    
    contentURL = ""
    
    #Check if there is an overlap of general festivals
    overlaps_gen_fest, festival_cult, gen_fest_url = festival_checker(date, read_festivals_from_file("./static/gen_fest.txt"))

    
    print(f"Willing: {has_extrovert(agent_behaviour=agent_behaviour, type='willing')}")
    print(f"Ready: {has_extrovert(agent_behaviour=agent_behaviour, type='ready')}")
    
    #If there is agents willing or ready to interact do below: 
    if((has_extrovert(agent_behaviour=agent_behaviour, type='willing')>= 2) and has_extrovert(agent_behaviour=agent_behaviour, type='ready')>=1):

        # CONTENT CHECKER
        print("******* CONTENT CHECKING ********")
        lang_index = -1
        
        #Check if there is a high similarity in interest value between 2 or more people    
        Ri_index, Li_index = extrovert_index(agent_behaviour=agent_behaviour)
        print("---- Similarity Index ----")
        print(f"Ri Index: {Ri_index}")
        print(f"Li Index: {Li_index}")
        
        # LANG CHECKER
        print("******* LANGUAGE CHECKING ********")
        lang_flag, lang_index = has_common(ready_index=Ri_index, willing_index=Li_index, subject_arr=agent_culture, start=0, end=4)
        
        print(f"Language Flag: {lang_flag}")
        print(f"Language Indexes: {lang_index}")
        
        print("******* INTEREST CHECKING ********")
        has_common_interest_flag, interest_index = has_common(ready_index=Ri_index, willing_index=Li_index, subject_arr=agent_interest)
        
        #IF yes do below:
        if(has_common_interest_flag):
            
            selected_lang = random.choice(lang_index)
            selected_int = random.choice(interest_index)
            
            print(f"Interest Index of {interest_index}, the value {selected_int} was selected")
            print(f"Language Index of {lang_index}, the value {selected_lang} was selected")
            
            #Select randomly what interest video to play
            contentURL=interest_vid_decider(interest_index=selected_int, lang_index=selected_lang, video_arr=read_interest_video_from_file("./static/interest_vid.txt"))
            
            
            
            
        
        #ELSE run a check on cultural since simiarity on interest failed
        else:
            print("******* RELIGIOUS CHECKING ********")
            #Get info on if has_common_flag, the common_religion, 
            has_common_rel_flag, religion_index = has_common(ready_index=Ri_index, willing_index=Li_index, subject_arr=agent_culture, start=8, end=12)

            print("******* CULT FEST DATE CHECKING ********")
            #Get info on cultural dates
            overlaps_cultural_fest, culture, cult_fest_url = festival_checker(date, read_festivals_from_file("./static/cult_fest.txt"))
            
        #   Check if there are similar cultures in group
        #   If got similar religious culture, do below  
            print(f"Has Common Religion Flag: {has_common_rel_flag}")
            print(f"Overlaps with date: {overlaps_cultural_fest}")  
            print(f"Religion Index: {religion_index}")
            print(f"Festival Religion: {culture}")    
            print(f"Matching culture and festival: {check_cult_fest_match(rel_arr=religion_index, fest_index=culture)}")     
            if(has_common_rel_flag and overlaps_cultural_fest and check_cult_fest_match(rel_arr=religion_index, fest_index=culture)):      
                contentURL=cult_fest_url
                
        #       IF religious festival is on the current date
        #           play video on cultural festival (3rd Option)
        #       ELSE
        #           
        #   ELSE Check for similar language since religious failed
            else:
                contentURL = "Generic Trend due to Failed Culture"
                if(overlaps_gen_fest):
                    # If got cultural festival
                    contentURL = gen_fest_url
        #       IF got similar language
        #           play video on trends with common language (4th Final Option)
                else:
                    #play genereic trend in general language (5th Final Option)
                    contentURL = "https://www.youtube.com/embed/tClfHvFqSdU"
        
    #ELSE run generic trends:
    else:
        if(overlaps_gen_fest):
            contentURL = gen_fest_url
        else:
            #   Run Generic trend in general language
            contentURL = "https://www.youtube.com/embed/iBQDveqtU"
     
    return contentURL   

# Database model for Agent
class Agent(Base):
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, index=True)
    Ha = Column(Float, default=0.5)
    Sd = Column(Float, default=0.5)
    Fe = Column(Float, default=0.5)
    Ex = Column(Float, default=0.5)
    Op = Column(Float, default=0.5)
    Nu = Column(Float, default=0.5)
    Eh = Column(Float, default=0.5)
    Hobb1 = Column(Integer, default=0)
    Hobb2 = Column(Integer, default=0)
    Hobb3 = Column(Integer, default=0)
    Hobb4 = Column(Integer, default=0)
    Hobb5 = Column(Integer, default=0)
    Hobb6 = Column(Integer, default=0)
    Int1 = Column(Integer, default=0)
    Int2 = Column(Integer, default=0)
    Int3 = Column(Integer, default=0)
    Int4 = Column(Integer, default=0)
    Int5 = Column(Integer, default=0)
    Int6 = Column(Integer, default=0)
    Lang1 = Column(Integer, default=0)
    Lang2 = Column(Integer, default=0)
    Lang3 = Column(Integer, default=0)
    Lang4 = Column(Integer, default=0)
    Race1 = Column(Integer, default=0)
    Race2 = Column(Integer, default=0)
    Race3 = Column(Integer, default=0)
    Race4 = Column(Integer, default=0)
    Rel1 = Column(Integer, default=0)
    Rel2 = Column(Integer, default=0)
    Rel3 = Column(Integer, default=0)
    Rel4 = Column(Integer, default=0)


# Create tables
Base.metadata.create_all(bind=engine)

#region Clustering Algorithm

#------------------------------------- Euclidian Distance Calculator
def euclid_dist(agentLoc, tvLoc):
    return np.sqrt((agentLoc[0] - tvLoc[0]) ** 2 + (agentLoc[1] - tvLoc[1]) ** 2)

#------------------------------------- Clustering Algorithm
def clustering(agent_array,agent_detail, agent_interest, agent_culture, tv_array):
    
    cluster = defaultdict(list)
    agent_packet = []
    
    # print(agent_array)
    # print(agent_detail)
    
    for index, agent in enumerate(agent_array):
        # print(index)
        # print(agent)
        closest_tv_index = None
        min_dist = float('inf')
        
        for tvindex, tv in enumerate(tv_array):
            dist = euclid_dist((agent[1], agent[2]), (tv[1], tv[2]))
            if dist < min_dist:
                min_dist= dist
                closest_tv_index = tvindex
                
        #Packet up the agent and its info
        agent_packet = [agent,agent_detail[index], agent_interest[index], agent_culture[index]]
        
        #Cluster is a 2d array where it saves it by groups of agents. [[A1, A2, A3],[A4,A5]]
        cluster[closest_tv_index].append(agent_packet)
        
        agent_packet=[]
        # print(index)
        # print(agent_detail[index])
        # cluster[closest_tv_index].append(agent_detail[index])
    
    
    # print("Cluster DEbug -------")
    # print(cluster)
    # print("Cluster DEbug -------")
        
    return cluster 
#endregion


#region Similarities

#-------------------------------------------------- Similarity AND Algo

def agent_similarity(Packets):      #Plan to pass both interest and cultural packets
    
    def sim(set1, set2):
        return 1 if any(a&b for a,b in zip(set1,set2)) else 0

    def OR_algo(similarity_matrix):
        # Perform a logical OR operation across each row
        return [1 if any(row) else 0 for row in similarity_matrix]
    
    print("---- Old Sim Algo ----")
    print("Input:")
    print(Packets)
    print("")
    
    similarities = np.zeros((len(Packets), len(Packets)))
    # print(f"Similarity intialisation: \n{similarities}")
    count_similar = []
    
    for i in range(len(Packets)):          #Excesses each agent detail in packet
        for j in range(i, len(Packets)):
            similarity = sim(Packets[i], Packets[j]) 
            similarities[i][j] = similarity
            similarities[j][i] = similarity
        # print(f"Similarity[i]: \n{similarities[i]}")
    
    #Count function
    count_similar =  [0] * len(Packets[0])  
    for array in Packets:
        count_similar = [a + b for a, b in zip(count_similar, array)]
    
    # print(f"Basic Avg: {average_similarities}")
    # print(f"Counted Array: {count_similar}")

        
    return OR_algo(similarities), count_similar


#endregion


#region SSIC Model
def run_ssic_model(agent_array):
    # Time and model parameters
    maxLimY = 1.2
    minLimX = 0
    numStep = GNumStep
    numStepChange = GNumStep
    dt = 0.1
    k = 12
    
    numAgents, numAttributes = agent_array.shape
    
    # Declare All Variables and Set INITIAL VALUES
    Pa = np.zeros((numAgents, numStep))
    Si = np.zeros((numAgents, numStep))
    Ri = np.zeros((numAgents, numStep))
    Dh = np.full((numAgents, numStep), 0.5)
    Ds = np.full((numAgents, numStep), 0.5)
    Df = np.full((numAgents, numStep), 0.5)
    Li = np.full((numAgents, numStep), 0.5)
    Psi = np.zeros((numAgents, numStep))

    # Initialisation parameters
    beta_Pa = 0.2
    omega_Ps = 0.5
    beta_Si = 0.5
    omega_Ri = 0.5
    beta_Ri = 1.0
    gamma_Dh = 0.1
    lambda_Dh = 0.03
    gamma_Ds = 0.1
    lambda_Ds = 0.03
    gamma_Df = 0.1
    lambda_Df = 0.03
    gamma_Li = 0.5
    
    #-------------------------------------------------------------
    # Run the model at t=1
    for i in range(numAgents):
        Pa[i, 0] = Dh[i, 0] - (beta_Pa * Ds[i, 0])
        Si[i, 0] = beta_Si * Pa[i, 0] + (1 - beta_Si) * (omega_Ps * agent_array[i, 3] + (1 - omega_Ps) * agent_array[i, 4]) * agent_array[i, 7] * (1 - agent_array[i, 6])
        Psi[i, 0] = 1 / (1 + np.exp(-k * (Df[i, 0] * agent_array[i, 5])))
        Ri[i, 0] = beta_Ri * (omega_Ri * Si[i, 0] + (1 - omega_Ri) * Li[i, 0]) * agent_array[i, 8] * (1 - Psi[i, 0])

    # Run the model at t=2
    for t in range(1, numStep):
        for i in range(numAgents):
            Pa[i, t] = Dh[i, t-1] - (beta_Pa * Ds[i, t])
            Si[i, t] = beta_Si * Pa[i, t] + (1 - beta_Si) * (omega_Ps * agent_array[i, 3] + (1 - omega_Ps) * agent_array[i, 4]) * agent_array[i, 7] * (1 - agent_array[i, 6])
            Psi[i, t] = Df[i, t-1] * agent_array[i, 5] / (1 + np.exp(-k * (Df[i, t-1] * agent_array[i, 5])))
            Ri[i, t] = beta_Ri * (omega_Ri * Si[i, t] + (1 - omega_Ri) * Li[i, t-1]) * agent_array[i, 8] * (1 - Psi[i, t])
            
            Dh[i, t] = Dh[i, t-1] + gamma_Dh * (agent_array[i, 0] - lambda_Dh) * Dh[i, t-1] * (1 - Dh[i, t-1]) * dt
            Ds[i, t] = Ds[i, t-1] + gamma_Ds * (agent_array[i, 1] - lambda_Ds) * Ds[i, t-1] * (1 - Ds[i, t-1]) * dt
            Df[i, t] = Df[i, t-1] + gamma_Df * (agent_array[i, 2] - lambda_Df) * Df[i, t-1] * (1 - Df[i, t-1]) * dt
            Li[i, t] = Li[i, t-1] + gamma_Li * (Si[i, t-1] - Li[i, t-1]) * (1 - Li[i, t-1]) * Li[i, t-1] * dt

    return Pa, Si, Ri, Dh, Ds, Df, Li, Psi  # or other desired outputs
#endregion


#region Pages Basic Functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# HTML form rendering
@app.get("/agent", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

# HTML form rendering for users
@app.get("/register", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("userregister.html", {"request": request})

# Add TV to session
# Endpoint to handle the TV addition form
@app.post("/add_tv")
async def add_tv(tv_x: int = Form(...), tv_y: int = Form(...)):
    # Generate a unique name based on the count of TVs
    tv_name = f"TV-{len(tv_positions) + 1}"
    tv_positions.append([tv_name, tv_x, tv_y])
    print(f"Current TV Positions: {tv_positions}")  # Debugging print statement
    
    return RedirectResponse(url="/", status_code=303)

# Endpoint to handle the Agent addition form
@app.post("/add_agent")
async def add_agent(name: str = Form(...), posX: int = Form(...), posY: int = Form(...), db: Session = Depends(get_db)):
    # Check if agent exists in DB
    agent = db.query(Agent).filter(Agent.name == name).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Add the agent details to the list
    agents_list.append([name, posX, posY])
    return RedirectResponse(url="/", status_code=303)

# Form submission to save data to the database
@app.post("/submit/")
async def submit_form(
    name: str = Form(...),
    #Personality
    Ex1R: int = Form(...),
    Ex6: int = Form(...),
    Op5R: int = Form(...),
    Op10: int = Form(...),
    Nu4R: int = Form(...),
    Nu9: int = Form(...),
    
    #Emotion
    Ha: float = Form(...),
    Sd: float = Form(...),
    Fe: float = Form(...),
    Eh: float = Form(...),
    
    #Hobb & Interest
    HobbArr: List[str] = Form([]),
    IntArr: List[str] = Form([]),
    
    #
    Language: List[str] = Form([]),
    Race: int = Form([]),
    Religion: int = Form([]),
    db: Session = Depends(get_db)
    
    ):
    
    # Calculate float values 
    ex1r_float = (6 - Ex1R) / 5 
    ex6_float = (Ex6 - 1) / 4 
    op5r_float = (6 - Op5R) / 5 
    op10_float = (Op10 - 1) / 4 
    nu4r_float = (6 - Nu4R) / 5 
    nu9_float = (Nu9 - 1) / 4 
    
    # Assign to Ex, Op, and Nu variables
    Ex = (ex1r_float + ex6_float) / 2 
    Op = (op5r_float + op10_float) / 2 
    Nu = (nu4r_float + nu9_float) / 2
        
    # Translate Religion and Culture
    race_values = translate(Race)
    rel_values = translate(Religion)
    
    # Debugging print statements
    print(f"Name: {name}")
    print(f"Personality: Ex={Ex}, Op={Op}, Nu={Nu}")
    print(f"Emotions: Ha={Ha}, Sd={Sd}, Fe={Fe}, Eh={Eh}")
    print(f"HobbArr (raw): {HobbArr}")
    print(f"IntArr (raw): {IntArr}")
    print(f"Language (raw): {Language}")
    print(f"Race (raw): {Race}")
    print(f"Religion (raw): {Religion}")
    
    # Convert form checkbox data to integer values, using string comparisons for presence
    hobb_values = [int(str(i) in HobbArr) for i in range(1, 7)]
    int_values = [int(str(i) in IntArr) for i in range(1, 7)]
    lang_values = [int(str(i) in Language) for i in range(1, 5)]
    # race_values = [int(str(i) in Race) for i in range(1, 5)]
    # rel_values = [int(str(i) in Religion) for i in range(1, 5)]
    
    print(f"Converted Hobbies: {hobb_values}")
    print(f"Converted Interests: {int_values}")
    print(f"Converted Languages: {lang_values}")
    print(f"Converted Races: {race_values}")
    print(f"Converted Religions: {rel_values}")
    
    # Create new agent instance
    agent = Agent(
        name=name,
        Ex=Ex,
        Op=Op,
        Nu=Nu,
        Ha=Ha,
        Sd=Sd,
        Fe=Fe,
        Eh=Eh,
        Hobb1=hobb_values[0], Hobb2=hobb_values[1], Hobb3=hobb_values[2], Hobb4=hobb_values[3],
        Hobb5=hobb_values[4], Hobb6=hobb_values[5],
        Int1=int_values[0], Int2=int_values[1], Int3=int_values[2], Int4=int_values[3],
        Int5=int_values[4], Int6=int_values[5],
        Lang1=lang_values[0], Lang2=lang_values[1], Lang3=lang_values[2], Lang4=lang_values[3],
        Race1=race_values[0], Race2=race_values[1], Race3=race_values[2], Race4=race_values[3],
        Rel1=rel_values[0], Rel2=rel_values[1], Rel3=rel_values[2], Rel4=rel_values[3]
    )

    # Save to database
    db.add(agent)
    db.commit()

    return RedirectResponse(url="/", status_code=303)



# Endpoint to render agents as a plot (same as before)
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def to_home():
    return RedirectResponse(url=f"/simhome", status_code=303)

@app.get("/simhome")
async def read_index(request: Request, db: Session = Depends(get_db)):
    
    db_agents = db.query(Agent).all()
    
    # Create Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    ax.grid(True)

    for agent in agents_list:
        name, posX, posY = agent  # Unpack name, posX, posY from the list
        ax.plot(posX, posY, 'bo')  # Blue circle for agent positions
        ax.annotate(name, (posX, posY), textcoords="offset points", xytext=(0, 5), ha='center')
        print(name)
        
    # Plot TV positions
    for pos in tv_positions:
        tv_name, tv_x, tv_y = pos  # Unpack name, x, y from each TV entry
        ax.plot(tv_x, tv_y, 'rs')  # Red square for TV positions
        ax.annotate(tv_name, (tv_x, tv_y), textcoords="offset points", xytext=(0, 5), ha='center')


    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()

    agent_names = [agent.name for agent in db_agents]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "plot_data": img_base64,
        "agent_names": agent_names,
        "curr_agents": agents_list,
        "tv_positions": tv_positions
        })
    
#endregion


#region SIMULATION
@app.post("/simulate", response_class=HTMLResponse)
async def simulate(request: Request, db: Session = Depends(get_db), date: str = Form(...)):
    
    print("")
    print("")
    print("")
    print("")
    print("--------------------------- SIMULATION START ---------------------------")
    print("")
    print("")
    print("")
    print("")

    simulation_agents_detail = []
    simulation_agents_name = []
    simulation_agents_interest = []
    simulation_agents_culture = []

    # Retrieve agent data (same as before)
    print("---- Retrieve Agent Data ----")
    for agent_name, agent_posX, agent_posY in agents_list:
        
        Ni = 0.5
        Nc = 0.5
        
        if agent_name == "Low Case":
            Ni = 0.1
            Nc = 0.1
        elif agent_name == "High Case":
            Ni = 0.9
            Nc = 0.9
        else:
            Ni = 0.5
            Nc = 0.5
        
        agent = db.query(Agent).filter(Agent.name == agent_name).first()
        if agent:
            
            #Get Agent SSIC Model Details
            simulation_agents_detail.append([
                agent.Ha, agent.Sd, agent.Fe, agent.Ex, agent.Op,
                agent.Nu, agent.Eh, Nc, Ni,
            ])
            
            #Get Agent Locations for Cluster
            simulation_agents_name.append([agent_name,
                agent_posX, agent_posY])
            
            #Get Agent Interest Similarity Details
            simulation_agents_interest.append([
                agent.Hobb1, agent.Hobb2, agent.Hobb3, agent.Hobb4, agent.Hobb5, agent.Hobb6, 
                agent.Int1, agent.Int2, agent.Int3, agent.Int4, agent.Int5, agent.Int6   
            ])
            
            #Get Agent Cultural Similarity Details
            simulation_agents_culture.append([
                agent.Lang1, agent.Lang2, agent.Lang3, agent.Lang4,
                agent.Race1, agent.Race2, agent.Race3, agent.Race4,
                agent.Rel1, agent.Rel2, agent.Rel3, agent.Rel4  
            ])
    
    print("---- Done Retrieve Data -----") 
    print("")       
    # print("--------------AGENT ARRAY---------------")
    # for agent in agents_list:
    #     print(agent)
    
    # for agent in simulation_agents_detail:
    #     print(agent)
        
    # for agent in simulation_agents_name:
    #     print(agent)
        
    # for agent in simulation_agents_interest:
    #     print(agent)
        
    # for agent in simulation_agents_culture:
    #     print(agent)
    # print("----------------------------------------")
    
    #Add the cluster loop here
    
    print("---- Clustering Start ----")
    
    clusterAgent = clustering(agent_array = simulation_agents_name, agent_detail = simulation_agents_detail, agent_interest = simulation_agents_interest, agent_culture = simulation_agents_culture, tv_array=tv_positions)      
    for tv, cluster in clusterAgent.items(): #[key1: [AgentPacket, AgentPacket], key2: [AgentPacket, AgentPacket]]
        # print("Printing Cluster")
        print("")
        print(f"Cluster {tv+1} :")
        for agentPacket in cluster: # AgentPacket = [AgentName&Loc, AgentDetails]
            for agentDetail in agentPacket:
                print(agentDetail)
        print("")
    
    print("---- Clustering Complete ----")  
    print("")   
        
        
    #region Running the SSIC Model based on Clusters    
    # Save graphs to static directory
    image_urls = []             # [ Image_URL_Cluster_1, Image_URL_Cluster_2 ... ] // Image_URL_Cluster_1 = [Image1, Image2, Image3, ..]
    
    #Average Values per Agent
    cluster_interest = []       # [ Interest_Cluster_1, Interest_Cluster_2, ... ] // Interest_Cluster_1 = []
    cluster_interest_count = []
    cluster_culture = []        # [ Culture_Cluster_1, Culture_Cluster_2, ... ] // Culture_Cluster_1 = []
    cluster_culture_count = []
    cluster_behaviour = []
    cluster_content = []        # [ Content_Cluster_1, Content_Cluster_2, ...] // Content_Cluster_1 = ["PAth_of_video.mp4"]
       
    
     
    #Runs Each Cluster Differently
    print("---- Running Clusters ----")
    print("")
    for tv, cluster in clusterAgent.items():
        print(f"---- Cluster {tv+1} ----")
        clusterName = tv+1
        
        #Agent packet unpacker
        agent_array=[]
        agent_name=[]
        agent_interest=[]
        agent_culture=[]
        
        #SSIC Model results pack
        agent_behaviour=[]
        
        # print(cluster)
        
        
        #Unpacks the Agent Packets into their arrays to be used for different methods
        print("---- Unpacking Agent Packet ----")
        for agentPacket in cluster:
            agent_name.append(agentPacket[0])              #agentPacket[0] = agent name, x, y
            agent_array.append(agentPacket[1])             #agentPacket[1] = SSIC Model input
            agent_interest.append(agentPacket[2])          #agentPacket[2] = Interest Similarity input
            agent_culture.append(agentPacket[3])           #agentPacket[3] = Cultural Similarity input
                
        # print(agent_name)
        # print(agent_array)    
        print("---- Unpacking Compelete ----")
        print("")   
        
        #-------------------------- Interest Similarities -----------------------------------------
        print("---- Similarity and Model Adjustment ----")
        # print("----------------Interest Similarities-----------------")
        or_sim, count_int = agent_similarity(agent_interest)
        cluster_interest.append(or_sim)
        cluster_interest_count.append(count_int)
        # print(cluster_interest)
        
        #-------------------------- Cultural Similarities -----------------------------------------
        
        # print("----------------Cultural Similarities-----------------")
        or_cult, count_cult = agent_similarity(agent_culture)
        cluster_culture.append(or_cult)
        cluster_culture_count.append(count_cult)
        # print(cluster_culture)
        
        #-------------------------- SSIC MODEL -----------------------------------------
        
        #adjust the agent_array by tweaking the Ni and Nc Values
        print("Attempt Asjustment on Ni and Nc")
        print(f"Agent array SSIC Model: {agent_array}")
        print(f"Clustuer culture: {cluster_culture}")
        print(f"Cluster Interest: {cluster_interest}")
        for index, agent in enumerate(agent_array):
            agent[7] = cluster_culture[tv][index]
            agent[8] = cluster_interest[tv][index]
        print(" END Attempt Asjustment on Ni and Nc")        
        #------------------------- Print out final Agent Array ------------------------
        print("#------------------------- Print out final Agent Array ------------------------#")
        print(agent_array)
        print("---- Similarity and Model Adjustment End ----")
        print("")   
        
        # Run the SSIC model
        print("--- Running the SSIC Model ----")
        agent_array = np.array(agent_array)
        Pa, Si, Ri, Dh, Ds, Df, Li, Psi = run_ssic_model(agent_array)
        print("---- End SSIC Model Run ----")
        print("")   
        
        # print("--------------------------------------------------------- Shape of Pa: ---------------------------------------------------------") 
        # numpy_pa = np.array(Pa)
        # print(numpy_pa.shape)
        # print("--------------------------------------------------------- Shape of Pa: ---------------------------------------------------------") 
        
        #-------------------------- Final value of Behaviour -----------------------------------------
        #Take out each behaviour Pa, Si, Ri... (which is in a [[],[],[],....] format into a [[-1],[-1], [-1], ...] to get all of the final values of each agent)
        
        #Then whack it together and then give it to cluster_behaviour
        print("---- Extracting Final SSIC Model Data ----")
        for index in range(len(Pa)):
            agent_behaviour.append([
                # Pa[index][-1],   #Pa  - Positive Affect
                # Si[index][-1],   #Si  - Short-Term Willingness to Interact
                Ri[index][-1],   #Ri  - Readiness to Interact
                # Dh[index][-1],   #Dh  - Dynamic Happiness
                # Ds[index][-1],   #Ds  - Dynamic Sadness
                # Df[index][-1],   #Df  - Dynamic Fear
                Li[index][-1],   #Li  - Long-Term Willingness to Interact
                # Psi[index][-1],   #Psi - Experienced Fear
            ])
        cluster_behaviour.append(agent_behaviour)
        print("---- Finish Extraction ----")
        print("")   
            
        # print("_____________________ AGENT BEHAVIOUR _________________________________")    
        # print(agent_behaviour) #Cluster Behaviour reached
        
        #Feed it to the if else gods
        
        print("____ Decision Model ____")
        print(f"Date: {date}")
        cluster_content.append(
            decision_model(date=date, 
                       agent_behaviour=agent_behaviour, 
                       agent_interest=agent_interest,
                       agent_culture=agent_culture,
                       interest_count=count_int, 
                       interest_arr=or_sim,
                       culture_count=count_cult,
                       culture_arr=or_cult)
            )
        print("---- Decision model finished ----")

        # Save graphs to static directory
        cluster_image_urls = []
        
        def save_fig_to_file_cluster(fig, name):
            path = STATIC_DIR / f"{name}.png"
            fig.savefig(path)
            cluster_image_urls.append(f"/static/{name}.png")
        
        def save_fig_to_file(fig, name):
            path = STATIC_DIR / f"{name}.png"
            fig.savefig(path)
            image_urls.append(f"/static/{name}.png")
        
        #region Generate 3D surface plots
        
        #Temporal Graphs
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('Temporal Factors (3D Surface Plots)')
        time = np.arange(GNumStep)
        agents = np.arange(len(agent_name))
        T, A = np.meshgrid(time, agents)

        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot_surface(T, A, Dh, cmap='viridis')
        ax1.set_title('Dynamic Happiness')
        ax1.set_xlabel('Time steps')
        ax1.set_ylabel('Agents')
        ax1.set_zlabel('Levels')

        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.plot_surface(T, A, Ds, cmap='plasma')
        ax2.set_title('Dynamic Sadness')
        ax2.set_xlabel('Time steps')
        ax2.set_ylabel('Agents')
        ax2.set_zlabel('Levels')

        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.plot_surface(T, A, Df, cmap='cividis')
        ax3.set_title('Dynamic Fear')
        ax3.set_xlabel('Time steps')
        ax3.set_ylabel('Agents')
        ax3.set_zlabel('Levels')

        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.plot_surface(T, A, Li, cmap='magma')
        ax4.set_title('Long-Term Willingness to Interact')
        ax4.set_xlabel('Time steps')
        ax4.set_ylabel('Agents')
        ax4.set_zlabel('Levels')

        plt.tight_layout()
        save_fig_to_file_cluster(fig, f"3d_temporal_factors{clusterName}")
        plt.close(fig)
        
        #Instantaneous Graphs
        
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('Instantaneous Factors (3D Surface Plots)')
        time = np.arange(GNumStep)
        agents = np.arange(len(agent_name))
        T, A = np.meshgrid(time, agents)

        # Plot Positive Affect
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot_surface(T, A, Pa, cmap='viridis')
        ax1.set_title('Positive Affect')
        ax1.set_xlabel('Time steps')
        ax1.set_ylabel('Agents')
        ax1.set_zlabel('Levels')

        # Plot Short-Term Willingness to Interact
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.plot_surface(T, A, Si, cmap='plasma')
        ax2.set_title('Short-Term Willingness to Interact')
        ax2.set_xlabel('Time steps')
        ax2.set_ylabel('Agents')
        ax2.set_zlabel('Levels')

        # Plot Experienced Fear
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.plot_surface(T, A, Psi, cmap='cividis')
        ax3.set_title('Experienced Fear')
        ax3.set_xlabel('Time steps')
        ax3.set_ylabel('Agents')
        ax3.set_zlabel('Levels')

        # Plot Readiness to Interact
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.plot_surface(T, A, Ri, cmap='magma')
        ax4.set_title('Readiness to Interact')
        ax4.set_xlabel('Time steps')
        ax4.set_ylabel('Agents')
        ax4.set_zlabel('Levels')

        plt.tight_layout()
        save_fig_to_file_cluster(fig, f"3d_instantaneous_factors{clusterName}")
        plt.close(fig)
            
        #endregion


        #region Generate 2D line plots
        
        #Temporal Graphs
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Temporal Factors (2D Line Plots)')
        for i in range(len(agent_name)):
            axes[0, 0].plot(time, Dh[i, :], label=f'{agent_name[i]}')
            axes[0, 1].plot(time, Ds[i, :], label=f'{agent_name[i]}')
            axes[1, 0].plot(time, Df[i, :], label=f'{agent_name[i]}')
            axes[1, 1].plot(time, Li[i, :], label=f'{agent_name[i]}')
        axes[0, 0].set_title('Dynamic Happiness')
        axes[0, 1].set_title('Dynamic Sadness')
        axes[1, 0].set_title('Dynamic Fear')
        axes[1, 1].set_title('Long-Term Willingness to Interact')
        
        # Add legends to each subplot
        axes[0, 0].legend(loc='best', fontsize=10)
        axes[0, 1].legend(loc='best', fontsize=10)
        axes[1, 0].legend(loc='best', fontsize=10)
        axes[1, 1].legend(loc='best', fontsize=10)

        plt.tight_layout()
        save_fig_to_file_cluster(fig, f"2d_temporal_factors{clusterName}")
        plt.close(fig)
        
        #Instantaneous Graphs
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Temporal Factors (2D Line Plots)')
        for i in range(len(agent_name)):
            axes[0, 0].plot(time, Pa[i, :], label=f'{agent_name[i]}')
            axes[0, 1].plot(time, Si[i, :], label=f'{agent_name[i]}')
            axes[1, 0].plot(time, Psi[i, :], label=f'{agent_name[i]}')
            axes[1, 1].plot(time, Ri[i, :], label=f'{agent_name[i]}')
        axes[0, 0].set_title('Positive Affect')
        axes[0, 1].set_title('Short Term Willingness to Interact')
        axes[1, 0].set_title('Experienced Fear')
        axes[1, 1].set_title('Readiness to Interact')
        
        # Add legends to each subplot
        axes[0, 0].legend(loc='best', fontsize=10)
        axes[0, 1].legend(loc='best', fontsize=10)
        axes[1, 0].legend(loc='best', fontsize=10)
        axes[1, 1].legend(loc='best', fontsize=10)

        plt.tight_layout()
        save_fig_to_file_cluster(fig, f"2d_instantaneous_factors{clusterName}")
        plt.close(fig)
        
        # print("IMAGE URLS FOR CLUSTER")
        # print(cluster_image_urls)
        image_urls.append(cluster_image_urls)
        #endregion
    
        print(f"---- Cluster {tv+1} END ----")
        print("")
        print("")
        print("")
    # image_urls.append(cluster_image_urls)
    print("---- All clusters Ran ----")
    print("")
    #endregion
    
    
    print(f"Cluster DEBUG:")
    print(f'Interest Array: {cluster_interest}')
    print(f'Interest Count: {cluster_interest_count}')
    print(f'Cultural Array: {cluster_interest}')
    print(f'Cultural count: {cluster_culture_count}')
    print(f"Model Output: {cluster_behaviour}")
    print(f"Cluster Conent Output: {cluster_content}")
    print("")
    print("")
    
        
        
    print("")
    print("")
    print("")
    print("")
    print("--------------------------- SIMULATION END ---------------------------")
    print("")
    print("")
    print("")
    print("")
    
    # Render the result.html template with the image URLs
    return templates.TemplateResponse("result.html", {"request": request, "image_urls": image_urls, "interests": cluster_interest, "culture": cluster_culture, "content": cluster_content})

#endregion

# Static files setup (optional, depending on where your CSS/JS resides)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
