import arcade as rkd
from arcade import ShapeElementList, color
import numpy as np
from numba import njit, jit, int8

WIDTH, HEIGHT = 512, 512
cells = np.zeros(shape=(64,64),dtype=np.int8)

class GameWin(rkd.Window):

    def __init__(self):
        super().__init__(WIDTH,HEIGHT,title="Conway's Game of Life")
        self.last_cell :tuple[int,int] = (-1,-1,)
        self.set_to :bool = False
        self.elapsed :float = 0.0
        self.lines :ShapeElementList = None
        self.mousedown=False

    def setup(self):
        self.lines = rkd.ShapeElementList()
        for i in range(0,512,8):
            line = rkd.create_line(i,0,i,HEIGHT,color.GRAY)
            self.lines.append(line)
            line = rkd.create_line(0,i,WIDTH,i, color.GRAY)
            self.lines.append(line)


    def on_draw(self):
        self.clear(color.WHITE) 

        squares = rkd.ShapeElementList()
        for x in range(64):
            for y in range(64):
                if cells[(x,y)]:
                    square = rkd.create_rectangle_filled(x*8.-4.,y*8.-4.,8.,8.,color.BLUE_GRAY)
                    squares.append(square)
        
        self.lines.draw()
        squares.draw()

    def on_update(self, delta_time: float):
        global cells
        if not self.mousedown: 
            self.elapsed += delta_time
            if self.elapsed > 0.1:
                self.elapsed = 0
                evolve(cells)

        return super().on_update(delta_time)
 
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        self.mousedown=True
        this_x_y=(x//8+1,y//8+1)
        self.set_to = not cells[this_x_y]
        cells[this_x_y] = self.set_to
        self.last_cell=this_x_y
        return super().on_mouse_press(x, y, button, modifiers)

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int):
        self.mousedown=False
        return super().on_mouse_release(x, y, button, modifiers)
 
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int):
        this_x_y = (x//8+1,y//8+1)
        if not this_x_y == self.last_cell: 
            cells[this_x_y] = self.set_to
            self.last_cell = this_x_y 
        return super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)


def main():
    game = GameWin()
    game.setup()
    init_glider(10,50)
    init_glider(12,52)
    init_glider(32,39)
    init_glider(32,42)
    rkd.run() # blocks apparently

def init_glider(x,y):
    cells[x+1, y+2]=1
    cells[x+2, y+1]=1
    cells[x:x+3, y]=1

@njit(parallel=True, fastmath=True)
def evolve(cells):
    counts = np.zeros_like(cells,dtype=np.int8)
    for x in range(1,63): # avoid borders
        for y in range(1,63): 
            counts[x][y] =  cells[x-1][y+1] + cells[x][y+1] + cells[x+1][y+1] + \
                            cells[x-1][y  ] +               + cells[x+1][y  ] + \
                            cells[x-1][y-1] + cells[x][y-1] + cells[x+1][y-1] 
    for x in range(1,63): 
        for y in range(1,63): 
            if cells[x][y]==1 :
                if (counts[x][y] < 2 or counts[x][y] > 3): 
                    cells[x][y]=0
            else:
                if counts[x][y]==3: 
                    cells[x][y]=1 


if __name__ == "__main__": main()
 