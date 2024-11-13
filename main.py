from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import matplotlib.pyplot as plt
import io
import base64
import os
# from Agent import Agent

app = FastAPI()

# Setting up Jinja2 templates and static files
templates = Jinja2Templates(directory="templates")

# Memory storage for agents and TVs (consider using a real database in production)
agents_list = []
television_list = []

# Model to receive agent data
class AgentForm(BaseModel):
    name: str
    agent_x: int = 7
    agent_y: int = 7

@app.get("/")
async def read_index(request: Request):
    # Create Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    ax.grid(True)

    # Plot existing agents (blue dots)
    for agent in agents_list:
        ax.plot(agent['posX'], agent['posY'], 'bo')

    # Plot televisions (green triangles)
    for tv in television_list:
        ax.plot(tv[0], tv[1], 'g^')

    # Convert the plot to PNG and then to base64 string for embedding in HTML
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()

    # Pass the list of agents to the template
    agent_names = [agent['name'] for agent in agents_list]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "plot_data": img_base64,
        "agent_names": agent_names,
        "agents": agents_list
    })

@app.post("/add_tv")
async def add_television(tv_x: int = Form(...), tv_y: int = Form(...)):
    if 0 <= tv_x <= 10 and 0 <= tv_y <= 10:
        television_list.append((tv_x, tv_y))
    return RedirectResponse(url="/", status_code=303)

@app.post("/add_agent")
async def add_agent(name: str = Form(...), agent_x: int = Form(7), agent_y: int = Form(7)):
    # Create and add agent
    agent_data = {
        "name": name,
        "posX": agent_x,
        "posY": agent_y,
        "Ha": 0.5,
        "Sd": 0.5,
        "Fe": 0.5,
        "Ex": 0.5,
        "Op": 0.5,
        "Nu": 0.5,
        "Eh": 0.5,
        "Nc": 0.5,
        "Ni": 0.5,
        "Dh": 0.5,
        "Ds": 0.5,
        "Df": 0.5,
        "Li": 0.5,
        "HobbArr": [None] * 7,
        "IntArr": [None] * 6,
        "LangArr": [None] * 4,
        "RaceArr": [None] * 4,
        "RelArr": [None] * 4,
    }
    agents_list.append(agent_data)
    return RedirectResponse(url="/", status_code=303)

# Static files setup (optional, depending on where your CSS/JS resides)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
