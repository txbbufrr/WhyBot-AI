
import heapq
import random
import requests
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="WhyBot AI | Robot Mission Control",
    page_icon="🤖",
    layout="wide",
)


# =========================================================
# FUTURISTIC UI
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #080d1a 0%, #101a2d 100%);
        color: #e6f1ff;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    h1, h2, h3 {
        color: #e6f1ff !important;
    }

    p, label, .stMarkdown {
        color: #c6d6ed;
    }

    .stButton > button {
        background: linear-gradient(90deg, #087eaa, #2563eb);
        color: white;
        border: 1px solid #39c6ff;
        border-radius: 10px;
        padding: 0.55rem 1rem;
        font-weight: 600;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #a5f3fc;
        box-shadow: 0 0 15px #0891b255;
        color: white;
    }

    [data-testid="stMetric"] {
        background: #101d33;
        border: 1px solid #243e60;
        padding: 16px;
        border-radius: 14px;
    }

    [data-testid="stSidebar"] {
        background: #0b1425;
        border-right: 1px solid #243e60;
    }

    hr {
        border-color: #243754;
    }

    .mission-card {
        background: #101d33;
        border: 1px solid #243e60;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .mission-label {
        color: #8ca9cc;
        font-size: 12px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .mission-value {
        color: #39c6ff;
        font-size: 20px;
        font-weight: bold;
        margin-top: 5px;
    }

    .grid-board {
        display: grid;
        grid-template-columns: repeat(10, minmax(0, 1fr));
        gap: 5px;
        width: 100%;
        padding: 12px;
        background: #0b1425;
        border: 1px solid #243e60;
        border-radius: 14px;
        box-sizing: border-box;
    }

    .grid-cell {
        aspect-ratio: 1 / 1;
        min-width: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 6px;
        background: #16243b;
        border: 1px solid #253b5a;
        color: #9cb6d6;
        font-size: clamp(10px, 2vw, 23px);
        overflow: hidden;
    }

    .cell-path {
        background: #123d53;
        border-color: #168bb5;
    }

    .cell-obstacle {
        background: #3a2336;
        border-color: #a53b66;
    }

    .cell-start {
        background: #123d32;
        border-color: #34d399;
    }

    .cell-goal {
        background: #49371b;
        border-color: #fbbf24;
    }

    .cell-robot {
        background: #075985;
        border: 2px solid #67e8f9;
        box-shadow: 0 0 12px #06b6d4aa;
    }

    .legend {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        padding: 10px 0;
        color: #c6d6ed;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CONSTANTS
# =========================================================
GRID_SIZE = 10
START = (0, 0)
GOAL = (9, 9)


# =========================================================
# SESSION STATE
# =========================================================
defaults = {
    "blocked": set(),
    "robot_step": 0,
    "simulation_started": False,
    "emergency": False,
    "explanation": "",
    "mission_status": "Ready to start",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# A* PATHFINDING ALGORITHM
# =========================================================
def heuristic(a, b):
    """Manhattan distance between two grid cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, goal, blocked):
    """Find the shortest valid path using A*."""
    if start in blocked or goal in blocked:
        return None

    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0, start))

    came_from = {}
    g_score = {start: 0}
    closed = set()

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        if current == goal:
            path = [current]

            while current in came_from:
                current = came_from[current]
                path.append(current)

            path.reverse()
            return path

        closed.add(current)

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            row, col = neighbor

            if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
                continue

            if neighbor in blocked or neighbor in closed:
                continue

            tentative_g = current_cost + 1

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                f_score = tentative_g + heuristic(neighbor, goal)

                heapq.heappush(
                    open_heap,
                    (f_score, tentative_g, neighbor),
                )

    return None


# =========================================================
# PREPARE THE CURRENT PATH
# =========================================================
current_path = astar(
    START,
    GOAL,
    st.session_state.blocked,
)

st.session_state.path = current_path

if current_path is None:
    st.session_state.simulation_started = False
    st.session_state.mission_status = "No valid path available"


# =========================================================
# PAGE HEADER
# =========================================================
st.title("🤖 WhyBot AI")

st.markdown(
    "### Autonomous Robot Navigation & Mission Control"
)

st.caption(
    "A* pathfinding • Live simulation • Local Gemma AI • Emergency controls"
)

st.divider()


# =========================================================
# SIDEBAR CONTROLS
# =========================================================
with st.sidebar:
    st.header("⚙️ Mission Controls")

    st.markdown("**Navigation settings**")
    st.write(f"Grid size: {GRID_SIZE} × {GRID_SIZE}")
    st.write(f"Start position: {START}")
    st.write(f"Destination: {GOAL}")

    st.divider()

    # Add random obstacles
    if st.button(
        "➕ Add Random Obstacles",
        use_container_width=True,
    ):
        candidates = [
            (r, c)
            for r in range(GRID_SIZE)
            for c in range(GRID_SIZE)
            if (r, c) not in (START, GOAL)
            and (r, c) not in st.session_state.blocked
        ]

        number_to_add = min(8, len(candidates))

        if number_to_add:
            st.session_state.blocked.update(
                random.sample(candidates, number_to_add)
            )

        st.session_state.robot_step = 0
        st.session_state.simulation_started = False
        st.session_state.emergency = False
        st.session_state.explanation = ""
        st.session_state.mission_status = "Map updated"
        st.rerun()

    # Reset map
    if st.button(
        "🧹 Reset Map",
        use_container_width=True,
    ):
        st.session_state.blocked = set()
        st.session_state.robot_step = 0
        st.session_state.simulation_started = False
        st.session_state.emergency = False
        st.session_state.explanation = ""
        st.session_state.mission_status = "Map reset"
        st.rerun()

    st.divider()

    st.markdown("**Local AI**")
    st.caption("Explanation powered by Ollama and Gemma.")


    # Explain mission with Groq
    if st.button(
        "🧠 Explain the Mission",
        use_container_width=True,
    ):
        path = st.session_state.path

        if not path:
            st.session_state.explanation = (
                "No valid route exists. Remove some obstacles and try again."
            )

        else:
            prompt = f"""
You are WhyBot AI, an assistant explaining robot navigation.

The robot starts at {START} and must reach {GOAL}.
The grid is {GRID_SIZE} by {GRID_SIZE}.
Blocked cells: {sorted(st.session_state.blocked)}
Calculated A* path: {path}
Path length: {len(path) - 1} moves.

Explain in simple language:
1. How the robot can reach its destination.
2. How A* pathfinding helps.
3. How obstacles affect the route.
4. What the path length means.

Keep the explanation clear and concise.
Use three or four short bullet points.
"""

            try:
                with st.spinner("WhyBot AI is analyzing the mission..."):
                    api_key = st.secrets["GROQ_API_KEY"]

                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "llama-3.3-70b-versatile",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                            "temperature": 0.3,
                            "max_tokens": 300,
                        },
                        timeout=60,
                    )

                    response.raise_for_status()
                    result = response.json()

                    st.session_state.explanation = (
                        result["choices"][0]["message"]["content"].strip()
                    )

                    if not st.session_state.explanation:
                        st.session_state.explanation = (
                            "The AI returned an empty explanation. Please try again."
                        )

            except KeyError:
                st.session_state.explanation = (
                    "Groq API key is missing. Check your Streamlit Secrets."
                )

            except requests.exceptions.ConnectionError:
                st.session_state.explanation = (
                    "Could not connect to Groq. Check your internet connection."
                )

            except requests.exceptions.Timeout:
                st.session_state.explanation = (
                    "The AI took too long to respond. Please try again."
                )

            except requests.exceptions.HTTPError as error:
                st.session_state.explanation = (
                    f"Groq API error: {error.response.status_code}. "
                    "Check your API key, model availability, and usage limits."
                )

            except (requests.exceptions.RequestException, ValueError, IndexError, TypeError, KeyError) as error:
                st.session_state.explanation = (
                    f"Could not get an AI explanation: {error}"
                )

    if st.session_state.explanation:
        st.markdown("### AI Explanation")
        st.write(st.session_state.explanation)
```


# =========================================================
# MISSION OVERVIEW
# =========================================================
path = st.session_state.path
path_length = len(path) - 1 if path else 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Grid Size", f"{GRID_SIZE} × {GRID_SIZE}")

with col2:
    st.metric("Obstacles", len(st.session_state.blocked))

with col3:
    st.metric("Path Length", f"{path_length} moves")

with col4:
    st.metric(
        "Mission Status",
        "Ready" if path else "Blocked",
    )

st.divider()


# =========================================================
# LIVE ROBOT GRID
# =========================================================
def render_robot_grid():
    """Render the grid using HTML and current robot position."""
    path = st.session_state.path
    path_cells = set(path) if path else set()

    step = st.session_state.robot_step

    if path and step < len(path):
        robot_position = path[step]
    else:
        robot_position = START

    html = ['<div class="grid-board">']

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cell = (row, col)
            classes = ["grid-cell"]
            symbol = ""

            if cell in st.session_state.blocked:
                classes.append("cell-obstacle")
                symbol = "⬛"

            elif cell == robot_position and path:
                classes.append("cell-robot")
                symbol = "🤖"

            elif cell == START:
                classes.append("cell-start")
                symbol = "🟢"

            elif cell == GOAL:
                classes.append("cell-goal")
                symbol = "🏁"

            elif cell in path_cells:
                classes.append("cell-path")
                symbol = "·"

            html.append(
                f'<div class="{" ".join(classes)}">{symbol}</div>'
            )

    html.append("</div>")

    st.markdown("".join(html), unsafe_allow_html=True)


# =========================================================
# MISSION BUTTONS
# =========================================================
button1, button2 = st.columns(2)

with button1:
    start_clicked = st.button(
        "▶️ Start / Resume",
        use_container_width=True,
    )

with button2:
    reset_clicked = st.button(
        "🔄 Reset Robot",
        use_container_width=True,
    )


if start_clicked:
    if st.session_state.path:
        if st.session_state.robot_step >= len(st.session_state.path) - 1:
            st.session_state.robot_step = 0

        st.session_state.emergency = False
        st.session_state.simulation_started = True
        st.session_state.mission_status = "Robot moving"
    else:
        st.session_state.simulation_started = False
        st.session_state.mission_status = "No valid route"


if reset_clicked:
    st.session_state.robot_step = 0
    st.session_state.simulation_started = False
    st.session_state.emergency = False
    st.session_state.mission_status = "Robot reset"

# =========================================================
# LIVE MISSION FRAGMENT
# =========================================================
@st.fragment(run_every=0.15)
def live_mission_tick():
    path = st.session_state.path
    stop_clicked = st.button(
        "🛑 EMERGENCY STOP",
        key="live_emergency_stop",
        use_container_width=True,
    )

    if stop_clicked:
        st.session_state.emergency = True
        st.session_state.simulation_started = False
        st.session_state.mission_status = "Emergency stop activated"
    if not path:
        st.error(
            "🚫 No valid route exists. Reset the map or remove obstacles."
        )
        return

    total_moves = len(path) - 1

    st.session_state.robot_step = min(
        st.session_state.robot_step,
        total_moves,
    )

    current_step = st.session_state.robot_step
    robot_position = path[current_step]

    # Draw the current robot position
    render_robot_grid()

    st.markdown(
        """
        <div class="legend">
            <span>🤖 Robot</span>
            <span>🟢 Start</span>
            <span>🏁 Destination</span>
            <span>⬛ Obstacle</span>
            <span>🔹 Planned path</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Live statistics
    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.metric(
            "Current Position",
            f"({robot_position[0]}, {robot_position[1]})",
        )

    with stat2:
        st.metric(
            "Moves Completed",
            f"{current_step} / {total_moves}",
        )

    with stat3:
        progress = (
            current_step / total_moves
            if total_moves > 0
            else 1.0
        )

        st.metric(
            "Mission Progress",
            f"{progress * 100:.0f}%",
        )

    st.progress(progress)

    # Mission status and animation
    if st.session_state.emergency:
        st.error("🛑 Emergency stop active. Robot is paused.")

    elif current_step >= total_moves:
        st.success("🏁 Destination reached! Mission complete.")
        st.session_state.simulation_started = False
        st.session_state.mission_status = "Destination reached"

    elif st.session_state.simulation_started:
        st.info(
            f"🟢 Robot moving from row {robot_position[0]}, "
            f"column {robot_position[1]}"
        )

        # Advance one cell per fragment refresh
        st.session_state.robot_step += 1

        if st.session_state.robot_step >= total_moves:
            st.session_state.robot_step = total_moves
            st.session_state.simulation_started = False
            st.session_state.mission_status = "Destination reached"

    else:
        st.info(
            f"⏸️ Robot paused at row {robot_position[0]}, "
            f"column {robot_position[1]}. Use Start / Resume to continue."
        )


live_mission_tick()


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "WhyBot AI • A* Autonomous Navigation • Local Ollama/Gemma Integration"
)
