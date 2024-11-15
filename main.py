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

def run_ssic_model(agent_array):
    # Time and model parameters
    maxLimY = 1.2
    minLimX = 0
    numStep = 1000
    numStepChange = 1000
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
    omega_Ri = 0.0
    beta_Ri = 1.0
    gamma_Dh = 0.1
    lambda_Dh = 0.03
    gamma_Ds = 0.1
    lambda_Ds = 0.03
    gamma_Df = 0.1
    lambda_Df = 0.03
    gamma_Li = 0.5
    
    # Initial state at t=1
    for i in range(numAgents):
        Pa[i, 0] = Dh[i, 0] - (beta_Pa * Ds[i, 0])
        Si[i, 0] = beta_Si * Pa[i, 0] + (1 - beta_Si) * (omega_Ps * agent_array[i, 3] + (1 - omega_Ps) * agent_array[i, 4]) * agent_array[i, 7] * (1 - agent_array[i, 6])
        Psi[i, 0] = 1 / (1 + np.exp(-k * (Df[i, 0] * agent_array[i, 5])))
        Ri[i, 0] = beta_Ri * (omega_Ri * Si[i, 0] + (1 - omega_Ri) * Li[i, 0]) * agent_array[i, 8] * (1 - Psi[i, 0])

    # Simulation for t=2 to numStep
    for t in range(1, numStep):
        for i in range(numAgents):
            Dh[i, t] = Dh[i, t-1] + gamma_Dh * (agent_array[i, 0] - lambda_Dh) * Dh[i, t-1] * (1 - Dh[i, t-1]) * dt
            Ds[i, t] = Ds[i, t-1] + gamma_Ds * (agent_array[i, 1] - lambda_Ds) * Ds[i, t-1] * (1 - Ds[i, t-1]) * dt
            Df[i, t] = Df[i, t-1] + gamma_Df * (agent_array[i, 2] - lambda_Df) * Df[i, t-1] * (1 - Df[i, t-1]) * dt

            Pa[i, t] = Dh[i, t] - (beta_Pa * Ds[i, t])
            Si[i, t] = beta_Si * Pa[i, t] + (1 - beta_Si) * (omega_Ps * agent_array[i, 3] + (1 - omega_Ps) * agent_array[i, 4]) * agent_array[i, 7] * (1 - agent_array[i, 6])
            Li[i, t] = Li[i, t-1] + gamma_Li * (Si[i, t-1] - Li[i, t-1]) * (1 - Li[i, t-1]) * Li[i, t-1] * dt
            Psi[i, t] = Df[i, t] * agent_array[i, 5] / (1 + np.exp(-k * (Df[i, t] * agent_array[i, 5])))
            Ri[i, t] = beta_Ri * (omega_Ri * Si[i, t] + (1 - omega_Ri) * Li[i, t]) * agent_array[i, 8] * (1 - Psi[i, t])

    return Pa, Si, Ri, Dh, Ds, Df, Li, Psi  # or other desired outputs


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
    Ex: float = Form(...),
    Op: float = Form(...),
    Nu: float = Form(...),
    Ha: float = Form(...),
    Sd: float = Form(...),
    Fe: float = Form(...),
    Eh: float = Form(...),
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
    
@app.post("/simulate", response_class=HTMLResponse)
async def simulate(request: Request, db: Session = Depends(get_db)):
    simulation_agents = []

    # Retrieve agent data (same as before)
    for agent_name, _, _ in agents_list:
        agent = db.query(Agent).filter(Agent.name == agent_name).first()
        if agent:
            simulation_agents.append([
                agent.Ha, agent.Sd, agent.Fe, agent.Ex, agent.Op,
                agent.Nu, agent.Eh, 0.5, 0.5  # Add defaults if needed
            ])

    # Convert to numpy array
    agent_array = np.array(simulation_agents)

    # Run the SSIC model
    Pa, Si, Ri, Dh, Ds, Df, Li, Psi = run_ssic_model(agent_array)

    # Save graphs to static directory
    image_urls = []
    def save_fig_to_file(fig, name):
        path = STATIC_DIR / f"{name}.png"
        fig.savefig(path)
        image_urls.append(f"/static/{name}.png")

    # Generate 3D surface plots
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Temporal Factors (3D Surface Plots)')
    time = np.arange(1000)
    agents = np.arange(len(simulation_agents))
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
    save_fig_to_file(fig, "3d_temporal_factors")
    plt.close(fig)

    # Generate 2D line plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Temporal Factors (2D Line Plots)')
    for i in range(len(simulation_agents)):
        axes[0, 0].plot(time, Dh[i, :], label=f'Agent {i+1}')
        axes[0, 1].plot(time, Ds[i, :], label=f'Agent {i+1}')
        axes[1, 0].plot(time, Df[i, :], label=f'Agent {i+1}')
        axes[1, 1].plot(time, Li[i, :], label=f'Agent {i+1}')
    axes[0, 0].set_title('Dynamic Happiness')
    axes[0, 1].set_title('Dynamic Sadness')
    axes[1, 0].set_title('Dynamic Fear')
    axes[1, 1].set_title('Long-Term Willingness to Interact')

    plt.tight_layout()
    save_fig_to_file(fig, "2d_temporal_factors")
    plt.close(fig)

    # Render the result.html template with the image URLs
    return templates.TemplateResponse("result.html", {"request": request, "image_urls": image_urls})

# Static files setup (optional, depending on where your CSS/JS resides)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
