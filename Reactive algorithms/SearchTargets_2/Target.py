class Target(object):
    """description of class"""
    def __init__(self,x_pos,y_pos):
        self.x_pos=x_pos
        self.y_pos=y_pos
    def __repr__(self):
        return "target "+str(self.x_pos)+" "+str(self.y_pos)
    def pos(self):
        return (x_pos,y_pos)


