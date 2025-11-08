import random
from typing import List, Tuple, Dict

from mesa import Agent, Model
from mesa.space import MultiGrid


class RoomCleaningAgent(Agent):
    """
    Agente de limpieza: intenta limpiar la celda actual; si ya está limpia,
    se mueve aleatoriamente a una celda vecina válida (Moore, radio=1).
    """
    def __init__(self, unique_id: int, model: "RoomCleaningModel", start_pos: Tuple[int, int]):
        # Evitar super().__init__(unique_id, model) por incompatibilidad con Mesa 3.x
        self.unique_id = unique_id
        self.model = model
        self.pos = None  # requerido por MultiGrid.place_agent
        self.movements = 0
        # Mesa usa coordenadas (x, y) donde x=columna, y=fila
        self.start_pos = start_pos

    def step(self):
        # Primero intenta limpiar la celda actual
        if self.model.is_dirty(self.pos):
            self.model.clean_cell(self.pos)
            return

        # Si ya estaba limpia, moverse a una vecina válida
        neighborhood = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=1
        )
        if neighborhood:
            new_pos = random.choice(neighborhood)
            self.model.grid.move_agent(self, new_pos)
            self.movements += 1


class RoomCleaningModel(Model):
    """
    Modelo de simulación de limpieza utilizando Mesa.
    No utiliza mesa.time (deprecated). En su lugar, se usa un bucle manual
    que invoca step() en cada agente.
    """
    def __init__(self, rows: int, cols: int, num_agents: int, dirty_percentage: float, max_time: int):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.max_time = max_time

        # Grid de Mesa: ancho=cols, alto=rows
        self.grid = MultiGrid(width=cols, height=rows, torus=False)

        # Malla de estados de celdas: 'clean' / 'dirty' indexada [fila][columna]
        self.cell_state: List[List[str]] = [['clean' for _ in range(cols)] for _ in range(rows)]
        self._initialize_dirty_cells(dirty_percentage)

        # Crear y colocar agentes; todos comienzan en (0,0) para replicar el notebook
        self.agent_list: List[RoomCleaningAgent] = []
        start_pos = (0, 0)  # (x, y) -> (col, fila)
        for i in range(num_agents):
            agent = RoomCleaningAgent(unique_id=i, model=self, start_pos=start_pos)
            self.agent_list.append(agent)
            self.grid.place_agent(agent, start_pos)

        # Métricas
        self.time_steps = 0
        self.clean_percentages: List[float] = []

    def _initialize_dirty_cells(self, dirty_percentage: float):
        total_cells = self.rows * self.cols
        num_dirty_cells = int(total_cells * dirty_percentage / 100)
        # Generar todas las coordenadas (fila, columna)
        all_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        dirty_cells = random.sample(all_cells, num_dirty_cells)
        for r, c in dirty_cells:
            self.cell_state[r][c] = 'dirty'

    def is_dirty(self, pos: Tuple[int, int]) -> bool:
        x, y = pos  # pos es (col, fila)
        return self.cell_state[y][x] == 'dirty'

    def clean_cell(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        if self.cell_state[y][x] == 'dirty':
            self.cell_state[y][x] = 'clean'
            return True
        return False

    def all_cleaned(self) -> bool:
        for r in range(self.rows):
            for c in range(self.cols):
                if self.cell_state[r][c] == 'dirty':
                    return False
        return True

    def run_step(self) -> bool:
        """
        Ejecuta un paso de simulación manual (sin mesa.time).
        Retorna False cuando se alcanza max_time o todas las celdas están limpias.
        """
        if self.time_steps >= self.max_time or self.all_cleaned():
            return False

        # Scheduler manual: iterar los agentes y llamar step()
        for agent in list(self.agent_list):
            agent.step()

        self.time_steps += 1
        self.clean_percentages.append(self.get_clean_percentage())
        return True

    def get_clean_percentage(self) -> float:
        total_cells = self.rows * self.cols
        clean_cells = sum(row.count('clean') for row in self.cell_state)
        return (clean_cells / total_cells) * 100.0

    def run(self) -> Dict[str, float]:
        """
        Ejecuta la simulación hasta terminar o llegar al tiempo máximo.
        Retorna métricas agregadas.
        """
        while self.run_step():
            pass

        total_movements = sum(a.movements for a in self.agent_list)
        return {
            'time_to_clean': self.time_steps,
            'percentage_clean': self.get_clean_percentage(),
            'total_movements': total_movements
        }


def run_experiments() -> List[Dict[str, float]]:
    """
    Ejecuta simulaciones variando el número de agentes, replicando los parámetros del notebook.
    """
    agent_counts = [1, 2, 4, 8, 16]
    room_rows = 10
    room_cols = 10
    dirty_percentage = 30
    max_time = 1000

    simulation_results: List[Dict[str, float]] = []
    for num_agents in agent_counts:
        print(f"Running simulation with {num_agents} agents...")
        model = RoomCleaningModel(
            rows=room_rows,
            cols=room_cols,
            num_agents=num_agents,
            dirty_percentage=dirty_percentage,
            max_time=max_time
        )
        results = model.run()
        results['num_agents'] = num_agents
        simulation_results.append(results)

    return simulation_results


if __name__ == "__main__":
    results = run_experiments()
    for r in results:
        print(r)