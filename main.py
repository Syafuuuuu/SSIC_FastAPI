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

# FastAPI app instance
app = FastAPI()

# Database setup
DATABASE_URL = "sqlite:///./agents.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Add a global array to store TV coordinates
tv_positions = []
agents_list = []


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
    Hobb7 = Column(Integer, default=0)
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


# Dependency to get the database session
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

# Add TV to session
# Endpoint to handle the TV addition form
@app.post("/add_tv")
async def add_tv(tv_x: int = Form(...), tv_y: int = Form(...)):
    # Add the new TV position to the array
    tv_positions.append([tv_x, tv_y])
    print(f"Current TV Positions: {tv_positions}")  # Debugging print statement

    # Redirect to the main page after adding the TV
    return RedirectResponse(url="/", status_code=303)

# Endpoint to handle the Agent addition form
@app.post("/add_agent")
async def add_agent(name: str = Form(...), posX: int = Form(...), posY: int = Form(...)):
    # Add the new agent details as [name, posX, posY]
    agents_list.append([name, posX, posY])
    print(f"Current Agent Positions: {agents_list}")  # Debugging print statement

    # Redirect to the main page after adding the Agent
    return RedirectResponse(url="/", status_code=303)

# Form submission to save data to the database
@app.post("/submit/")
async def submit_form(
    name: str = Form(...),
    Ex: float = Form(...),
    Op: float = Form(...),
    Nu: float = Form(...),
    Ha: float = Form(...),
    Sd: float = Form(...),
    Fe: float = Form(...),
    HobbArr: List[str] = Form([]),
    IntArr: List[str] = Form([]),
    Language: List[str] = Form([]),
    Race: List[str] = Form([]),
    Religion: List[str] = Form([]),
    db: Session = Depends(get_db)
    
    ):
    
    # Debugging print statements
    print(f"Name: {name}")
    print(f"Personality: Ex={Ex}, Op={Op}, Nu={Nu}")
    print(f"Emotions: Ha={Ha}, Sd={Sd}, Fe={Fe}")
    print(f"HobbArr (raw): {HobbArr}")
    print(f"IntArr (raw): {IntArr}")
    print(f"Language (raw): {Language}")
    print(f"Race (raw): {Race}")
    print(f"Religion (raw): {Religion}")
    
    # Convert form checkbox data to integer values, using string comparisons for presence
    hobb_values = [int(str(i) in HobbArr) for i in range(1, 7)]
    int_values = [int(str(i) in IntArr) for i in range(1, 7)]
    lang_values = [int(str(i) in Language) for i in range(1, 5)]
    race_values = [int(str(i) in Race) for i in range(1, 5)]
    rel_values = [int(str(i) in Religion) for i in range(1, 5)]
    
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
        
    # Plot TV positions
    for pos in tv_positions:
        ax.plot(pos[0], pos[1], 'rs')  # Red square for TV positions

    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()

    agent_names = [agent.name for agent in db_agents]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "plot_data": img_base64,
        "agent_names": agent_names,
        "agents": agents_list,
        "tv_positions": tv_positions
    })

# Static files setup (optional, depending on where your CSS/JS resides)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
